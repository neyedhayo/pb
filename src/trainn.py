import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent)) #to remove bug

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from transformers import ViTForImageClassification

from src.opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import tracemalloc

from src.config import NUM_CLASSES, HEIGHT_WIDTH, NUM_WORKERS, AUGMENT, SEED
from src.local_utility import LightningModel, AlzheimerDataModule, predict_and_plot, set_seed
from src.tracker import track_emissions

def trainn_model(experiment_name: str = "CNN Baseline", base_type: str = None, augmentation=AUGMENT):
    efficency_tracker = track_emissions(experiment_name=experiment_name)

    @efficency_tracker
    def _run(config=None):
        model_name = f"{experiment_name}--{augmentation.capitalize()}"

        tracemalloc.start()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        augment_data = augmentation.lower() == "augmented"
        dm = AlzheimerDataModule(
            batch_size=config["batch_size"],
            height_width=HEIGHT_WIDTH,
            num_workers=NUM_WORKERS,
            augment_data=augment_data
        )
        dm.setup()

        L.pytorch.seed_everything(SEED)
        set_seed(SEED)

        if base_type.lower() == "cnn":
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(in_features=model.fc.in_features, out_features=NUM_CLASSES)

            if not ModuleValidator.is_valid(model):
                model = ModuleValidator.fix(model)
        else:
            model = ViTForImageClassification.from_pretrained(
                "google/vit-base-patch16-224-in21k", num_labels=NUM_CLASSES
            )

        lightning_model = LightningModel(model, learning_rate=float(config["learning_rate"]), base=base_type)

        callback = [
            ModelCheckpoint(save_top_k=1, mode="max", monitor="val_acc"),
        ]

        trainer = L.Trainer(
            max_epochs=config["epochs"],
            accelerator="gpu",
            devices="auto",
            logger=CSVLogger(save_dir="../logs/models", name=f"{base_type.lower()}-model"),
            deterministic=True,
            callbacks=callback,
            precision="32-true"
        )

        
        if base_type.lower() == "cnn":
            train_loader = dm.train_dataloader()
            optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))

            privacy_engine = PrivacyEngine(accountant="rdp")
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=1.2,  # Choose this based on privacy/utility needs
                max_grad_norm=1.0,
            )

            lightning_model.model = model
            lightning_model.dp_optimizer = optimizer

        trainer.fit(model=lightning_model, datamodule=dm)

        if base_type.lower() == "cnn":
            epsilon = privacy_engine.get_epsilon(delta=1e-5)
            print(f"Final Privacy Budget (ε): {epsilon:.4f}")

        predict_and_plot(model=lightning_model, trainer=trainer,
                         datamodule=dm, model_name=model_name)

        return dm, trainer, lightning_model

    return _run()