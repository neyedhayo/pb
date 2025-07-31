import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # to remove import bug

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from transformers import ViTForImageClassification

from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader
from opacus.validators import ModuleValidator

import wandb

from src.config import NUM_CLASSES, HEIGHT_WIDTH, NUM_WORKERS, AUGMENT, SEED, WANDB_ENTITY, WANDB_PROJECT, ExperimentName
from src.local_utility import LightningModel, MedicalImageDataModule, predict_and_plot, set_seed
from src.tracker import track_emissions

def traindp_model(data_name: str, experiment_name: ExperimentName = ExperimentName.DP_CNN, base_type: str = None, augmentation=AUGMENT, num_labels: int = NUM_CLASSES):
    """
    Train and evaluate a (DP-)wrapped CNN or ViT model with manual DPDataLoader wrapping.

    Returns:
        dm: Lightning DataModule
        trainer: Lightning Trainer
        lightning_model: Wrapped model for training/evaluation
        privacy_engine: Configured Opacus PrivacyEngine to query ε post-training
    """
    efficency_tracker = track_emissions(experiment_name=experiment_name.value, data_name=data_name)

    @efficency_tracker
    def _run(config=None):
        model_name = f"{experiment_name.value}--{data_name}"

        # Memory tracking
        import tracemalloc
        tracemalloc.start()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # DataModule setup
        dm = MedicalImageDataModule(
            data_name= data_name,
            batch_size=config["batch_size"],
            height_width=HEIGHT_WIDTH,
            num_workers=NUM_WORKERS,
            augment_data=augmentation
        )
        dm.setup()

        # Reproducibility
        L.pytorch.seed_everything(SEED)
        set_seed(SEED)

        # Model initialization
        if base_type.lower() == "cnn":
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
            if not ModuleValidator.is_valid(model):
                model = ModuleValidator.fix(model)
        else:
            model = ViTForImageClassification.from_pretrained(
                "google/vit-base-patch16-224-in21k", num_labels=NUM_CLASSES
            )

        # Lightning wrapper
        lightning_model = LightningModel(
            model,
            learning_rate=float(config["learning_rate"]),
            base=base_type, 
            num_labels= num_labels
        )

        # Checkpoint callback & logger
        callbacks = [
            ModelCheckpoint(save_top_k=1, mode="max", monitor="val_acc"),
            EarlyStopping(monitor="val_loss", patience=config.get('tolerance'), mode="min")
            
        ]
        
        loggers = [
            CSVLogger(save_dir="../../logs/models", name=f"{base_type.lower()}-model"),
            WandbLogger(
                save_dir = "../../logs/wandb",
                project = WANDB_PROJECT,
                entity = WANDB_ENTITY,
                name = f"{experiment_name.value}_{data_name}"
            )
        ]

        # Trainer config
        trainer = L.Trainer(
            max_epochs=config["epochs"],
            accelerator="gpu",
            devices=1,
            logger=loggers,
            deterministic=True,
            callbacks=callbacks,
            precision="32-true",
        )

        privacy_engine = None
        
        # Only wrap DP logic for CNN
        if base_type.lower() == "cnn":
            
            # Use base DataLoader
            base_loader = dm.train_dataloader()

            # Wrap with DPDataLoader for Poisson subsampling
            dp_loader = DPDataLoader.from_data_loader(base_loader)

            # 3) Optimizer and Privacy Engine
            optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
            privacy_engine = PrivacyEngine(accountant="rdp")
            
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=dp_loader,
                noise_multiplier=1.0,   #ε
                max_grad_norm=1.0,
                poisson_sampling=False,
                clipping_per_layer=True, #0.5
            )

            # Inject DP-wrapped components back
            lightning_model.model = model
            lightning_model.dp_optimizer = optimizer
            dm.train_dataloader = lambda: train_loader

        # Train & validate
        trainer.fit(model=lightning_model, datamodule=dm)

        # Report privacy budget
        if base_type.lower() == "cnn" and privacy_engine is not None:
            epsilon = privacy_engine.get_epsilon(delta=1e-5)
            print(f"Final Privacy Budget (ε): {epsilon:.4f}")

        # Evaluation
        predict_and_plot(
            model=lightning_model,
            trainer=trainer,
            datamodule=dm,
            num_labels= num_labels,
            model_name=model_name, 
            experiment_name=experiment_name
        )
        
        wandb.finish()
        return dm, trainer, lightning_model

    return _run()