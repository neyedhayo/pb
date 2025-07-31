"MAIN: Training a Differentially Private (DP) model with Opcaus Privacy Engine"

from pathlib import Path
import time
import logging
from codecarbon import EmissionsTracker

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, ResNet
from torchvision.models import resnet18, ResNet18_Weights
from transformers import ViTForImageClassification

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader
from opacus.validators import ModuleValidator

from src.config import NUM_CLASSES, HEIGHT_WIDTH, NUM_WORKERS, AUGMENT, SEED, WANDB_ENTITY, WANDB_PROJECT, ExperimentName
from src.local_utility import LightningModel, MedicalImageDataModule, predict_and_plot, set_seed, set_device, load_yaml_config
from src.tracker import track_emissions, setup_wandb, start_memory_tracking, get_peak_memory_usage, append_peak_memory
from src.federated import evaluate_model

DEVICE = set_device()

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')





# ------------------------------ Model Utilities ------------------------------

class SafeBasicBlock(BasicBlock):
    """
    Custom ResNet BasicBlock that avoids in-place additions,
    which are incompatible with Opacus backward hooks.
    """
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity  # Avoid in-place `+=`
        out = self.relu(out)
        return out


def make_fully_opacus_safe_resnet18(num_classes: int) -> nn.Module:
    """
    Creates an Opacus-safe ResNet18 using custom BasicBlock.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        nn.Module: Modified ResNet18 model safe for DP training.
    """
    model = ResNet(block=SafeBasicBlock, layers=[2, 2, 2, 2])
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    for _, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False  # Avoid in-place ReLU

    return model



# -------------------------- Centralized DP Training --------------------------

def train_model(data_name: str, experiment_name: ExperimentName, base_type: str = None, augmentation=AUGMENT, num_labels: int = None):
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
            model = make_fully_opacus_safe_resnet18(NUM_CLASSES)
        else:
            model = ViTForImageClassification.from_pretrained(
                "google/vit-base-patch16-224-in21k", num_labels=NUM_CLASSES
            )
            
        if not ModuleValidator.is_valid(model):
            model = ModuleValidator.fix(model)

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
        
            
        # Use base DataLoader
        base_loader = dm.train_dataloader()

        # Wrap with DPDataLoader for Poisson subsampling
        dp_loader = DPDataLoader.from_data_loader(base_loader)

        # Optimizer and Privacy Engine
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
        privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=False)
        
        
        model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=dp_loader,
                noise_multiplier=config['dp_params'].get('epsilon')[0], #<--- [0] [1] = 0.1, 1.0
                # noise_multiplier=config['dp_params'].get('epsilon'),
                max_grad_norm=config['dp_params'].get('max_grad_norm'),
                poisson_sampling=config['dp_params'].get('poisson_sampling'),
                clipping_per_layer=config['dp_params'].get('clipping_per_layer'),
            )
        
        # Inject DP-wrapped components back
        lightning_model.model = model
        lightning_model.dp_optimizer = optimizer
        dm.train_dataloader = lambda: train_loader

        # Train & validate
        trainer.fit(model=lightning_model, datamodule=dm)

        # Report privacy budget
        if privacy_engine is not None:
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
        
        return dm, trainer, lightning_model

    return _run()



# ------------------------ Federated DP Training Setup ------------------------


track_config = load_yaml_config(key="tracker")

def train_FL_model(model, train_loader, val_loader, client_id, base_type, fed_config, target_delta, noise_multiplier, max_grad_norm, output_dir, optimizer=None):
    
    set_seed(seed_torch=True)
    
    tracker = EmissionsTracker(
        project_name=f"Client {client_id}",
        output_dir=Path(output_dir),
        output_file=track_config.get("client_output_file"),
        allow_multiple_runs=track_config.get("allow_multiple_runs", True),
        log_level=track_config.get("log_level"),
        measure_power_secs=track_config.get("measure_power_secs"),
        save_to_file=True
    )

    start_memory_tracking()
    tracker.start()
        
    if not ModuleValidator.is_valid(model):
            model = ModuleValidator.fix(model)
            
    criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(fed_config['learning_rate']))

    privacy_engine = PrivacyEngine(secure_mode=False)
    model.train()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    model.to(DEVICE)

    stats = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    # Early stopping variables
    patience_level = fed_config.get("tolerance", 7)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None
    
    try:
        start_time = time.perf_counter()

        for epoch in range(fed_config["epochs"]):
            model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            for batch_idx, (features, labels) in enumerate(train_loader):
                features, labels = features.to(DEVICE), labels.to(DEVICE)

                logits = model(features) if base_type == "cnn" else model(features).logits
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                train_correct += torch.sum(predictions == labels).item()
                train_total += len(labels)

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            stats['train_loss'].append(train_loss)
            stats['train_acc'].append(train_acc)

            val_loss, val_acc = evaluate_model(model, val_loader, base_type=base_type)
            stats['val_loss'].append(val_loss)
            stats['val_acc'].append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience_level:
                    print(f"Early stopping at epoch {epoch+1} (no improvement in {patience_level} epochs)")
                    break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        epsilon = privacy_engine.get_epsilon(delta=float(target_delta))

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        epsilon = None

    finally:
        end_time = time.perf_counter()
        total_time = end_time - start_time
        tracker.stop()
        
        cpu_peak, gpu_peak = get_peak_memory_usage()
        append_peak_memory(
            name=f"Client {client_id}",
            cpu_mem=cpu_peak,
            gpu_mem=gpu_peak,
            source="client",
            file_path=track_config.get('base_mem_dir'),
            duration=total_time
            )

    print(f"\n🔎 Tracker: {fed_config['name']}")
    print(f"📁 Logs saved to: {output_dir}/{track_config.get('client_output_file')}")
    print(f"⏱️ Total training time: {total_time//60:.0f} minutes {total_time % 60:.0f} seconds")

    return model, stats, epsilon

