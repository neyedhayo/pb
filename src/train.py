import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

import tracemalloc

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from transformers import ViTForImageClassification
 
from src.config import NUM_CLASSES, HEIGHT_WIDTH, NUM_WORKERS, AUGMENT, SEED, WANDB_ENTITY, WANDB_PROJECT, ExperimentName
from src.local_utility import LightningModel, MedicalImageDataModule, predict_and_plot, set_seed
from src.tracker import track_emissions



def train_model(data_name: str, experiment_name: ExperimentName = ExperimentName.CNN_BASE, base_type: str = None, augmentation=AUGMENT, num_labels: int = NUM_CLASSES):
    """
    Trains an Medical Image's disease classification model with optional augmentation.

    This function handles model training using PyTorch Lightning, tracks energy usage and 
    carbon emissions via CodeCarbon, and supports different experimental configurations including 
    Federated Learning, Differential Privacy, and Secure Computation techniques. Training 
    configurations are loaded from a central YAML file based on the provided experiment name.

    Args:
        experiment_name (str, optional): Name of the experiment configuration to load from 
            the `experiments.yaml` file. Determines hyperparameters like learning rate, 
            batch size, number of epochs, and training strategies.
            Examples:
                - "CNN Baseline"
                - "ViT Baseline", etc.

        base_type (str, optional): Name of the base model architecture to use in model training. If `None`, 
        ViT will be assumed as the base model architecture of choice.
            Options:
                - "vit": Vision Transformer
                - 'cnn": Convolutional Neural Network

        augmentation (str, optional): Specifies the data augmentation strategy to use.
            Options:
                - "original": Use raw/resized images without additional augmentations.
                - "augmented": Apply random cropping, flipping, and color jittering.
        
        num_labels (int, optional): The number of class label in selected dataset.

    Returns:
        Callable: A wrapped function that, when invoked, trains the model and returns:
            - dm (MedicalImageDataModule): The PyTorch Lightning data module used for loading data.
            - trainer (L.Trainer): The Lightning Trainer object used to fit the model.
            - lightning_model (LightningModel): The trained Lightning model instance.
    """
    efficency_tracker = track_emissions(experiment_name=experiment_name.value, data_name=data_name)

    @efficency_tracker
    def _run(config=None):
        model_name = f"{experiment_name.value}--{data_name}"
        
        # Initialize memory tracking
        tracemalloc.start()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        augment_data = augmentation
        dm = MedicalImageDataModule(
            data_name = data_name,
            batch_size=config["batch_size"],
            height_width=HEIGHT_WIDTH,
            num_workers=NUM_WORKERS,
            augment_data=augment_data
        )
        dm.setup()

        L.pytorch.seed_everything(SEED)
        set_seed(SEED)
        
        if base_type.lower() == "cnn":
            # Use ResNet18 Pretrained
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_labels)
        else:
            # Use ViT Pretrained
            model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=num_labels)

        lightning_model = LightningModel(model, learning_rate=float(config["learning_rate"]), base=base_type, num_labels=num_labels)

        callback = [
            ModelCheckpoint(save_top_k=1, mode="max", monitor="val_acc"),
            EarlyStopping(monitor="val_loss", patience=config.get('tolerance'), mode="min")
        ]
        
        csv_logger = CSVLogger(save_dir="../../logs/models", name=f"{base_type.lower()}-model")
        wandb_logger = WandbLogger(
            save_dir = "../../logs/wandb",
            project = WANDB_PROJECT,
            entity = WANDB_ENTITY,
            name = f"{experiment_name.value}_{data_name}"
        )

        trainer = L.Trainer(
            max_epochs=config["epochs"],
            accelerator="gpu",
            devices="auto",
            logger=[csv_logger, wandb_logger],
            deterministic=True,
            callbacks=callback,
            precision="16-mixed"
        )
        
        trainer.fit(model=lightning_model, datamodule=dm)
        predict_and_plot(model=lightning_model, trainer=trainer,datamodule=dm, num_labels = num_labels, model_name=model_name, experiment_name=experiment_name.value)
        
        return dm, trainer, lightning_model

    return _run()
