import os
import numpy as np
import pandas as pd
import wandb
import yaml
import random
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
import lightning as L

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torchmetrics
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, ROC, MatthewsCorrCoef, ConfusionMatrix
from sklearn.metrics import classification_report

import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import DatasetDict, load_dataset
from collections import OrderedDict
from typing import List, Optional, Literal

from src.paths import SOURCE_DIR
from src import config
from src.config import NUM_WORKERS, SEED, ExperimentName



# ---------------------------------- ACCELERATOR SETUP & CONTROL-FOR-RANDOMNESS ---------------------------------------


def set_device() -> str:
    """
    Determine the available computing device (GPU or CPU).

    This function checks if a CUDA-compatible GPU is available. If so, 
    it returns "cuda", otherwise, it defaults to "cpu".

    Returns:
        str: The name of the computing device, either "cuda" (GPU) or "cpu".

    Example:
        >>> set_device()
        'cuda'  # If GPU is available
        'cpu'   # If GPU is not available
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def set_seed(seed: int = SEED, seed_torch: bool = True):
    """
    Seeds the random number generators of PyTorch, Lightning, NumPy, and Python's `random` module to ensure
    reproducibility of results across runs when using PyTorch for deep learning experiments.

    This function sets the seed for PyTorch (both CPU and CUDA), Lightning, NumPy, and the Python `random` module,
    enabling CuDNN benchmarking and deterministic algorithms. It is crucial for experiments requiring
    reproducibility, like model performance comparisons. Note that enabling CuDNN benchmarking and
    deterministic operations may impact performance and limit certain optimizations.

    Args:
        seed (int, optional):
            A non-negative integer that defines the random state. Defaults to 'SEED' value in config file.

        seed_torch (bool, optional): 
            If `True` sets the random seed for pytorch tensors, so pytorch module
            must be imported. Defaults to True.
    Returns:
        None
            This function does not return a value but sets the random seed for various libraries.

    Notes:
        - When using multiple GPUs, `th.cuda.manual_seed_all(seed)` ensures all GPUs are seeded, 
        crucial for reproducibility in multi-GPU setups.

    Example:
        >>> SEED = 42
        >>> set_seed(SEED)
    """
    random.seed(seed)
    np.random.seed(seed)

    if seed_torch:
        L.pytorch.seed_everything(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# ---------------------------------- LOAD YAML CONFIG ---------------------------------------


def load_yaml_config(yaml_path: Path = f"{SOURCE_DIR}/experiments.yaml", key: str = None, item_name: str = None) -> dict:
    """
    Loads a named configuration item from a specified section in a YAML file.

    Args:
        yaml_path: Path to the YAML file.
        key: The top-level key in the YAML file (e.g., "experiments", "tracker").
        item_name: The name field to match within the section (optional).
                   If None, and the section is not a list, returns the section directly.

    Returns:
        dict: The matched config item, or the full section if item_name is None.

    Raises:
        FileNotFoundError: If YAML file does not exist.
        ValueError: If required arguments are missing or item is not found.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found at {yaml_path}")

    with open(yaml_path, mode="r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if key is None:
        raise ValueError("You must specify a top-level key to load (e.g., 'experiments', 'tracker').")

    section = config.get(key)
    if section is None:
        raise ValueError(f"No section '{key}' found in YAML.")

    if isinstance(section, list):
        if item_name is None:
            raise ValueError(f"Must provide item_name to search within list under '{key}'.")
        normalized_item_name = item_name.lower().replace(" ", "_")
        for item_dict in section:
            name = item_dict.get("name")
            if name and name.lower().replace(" ", "_") == normalized_item_name:
                return item_dict
        raise ValueError(f"'{item_name}' not found in section '{key}'.")
    elif isinstance(section, dict):
        if item_name is not None:
            raise ValueError(f"Cannot use item_name when section '{key}' is not a list.")
        return section
    else:
        raise ValueError(f"Unsupported format for section '{key}'.")


# ---------------------------------- DATA TRANSFORM & AUGMENTATION ------------------------------------
    
    
def get_transforms(data_name: str = "alzheimer", height_width = config.HEIGHT_WIDTH, augment: bool = True):
    """
    Returns training and validation transformations based on dataset name.
    If data_name is 'skin_lesions', use heavy albumentations. Otherwise, basic torchvision.
    """
    if data_name == "skin_lesions":
        if augment:
            train_transform = A.Compose([
                A.Resize(config.RESIZE_IMAGE[0], config.RESIZE_IMAGE[1]),
                A.RandomScale(config.A_RANDOM_SCALE),
                A.RandomCrop(height_width[0], height_width[1]),
                A.Rotate(limit=config.A_ROTATE_LIMIT),
                A.RandomBrightnessContrast(**config.A_BRIGHTNESS_CONTRAST),
                A.HorizontalFlip(p=config.A_FLIP_PROB),
                A.Affine(shear=config.A_SHEAR),
                
                A.CoarseDropout(
                    num_holes_range=(1, random.randint(1, config.A_COARSE_MAX_HOLES)),
                    hole_height_range=(config.A_COARSE_HEIGHT, config.A_COARSE_WIDTH),
                    hole_width_range=(config.A_COARSE_HEIGHT, config.A_COARSE_WIDTH),
                    p=config.A_COARSE_P
                    ),
                A.Normalize(),
                ToTensorV2(),
            ])
        else:
            train_transform = A.Compose([
                A.CenterCrop(height_width[0], height_width[1]),
                A.Normalize(),
                ToTensorV2(),
            ])

        test_transform = A.Compose([
            A.CenterCrop(height_width[0], height_width[1]),
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        # Default for Alzheimer
        if augment:
            train_transform = transforms.Compose([
                transforms.Resize(config.RESIZE_IMAGE),
                transforms.RandomCrop(config.HEIGHT_WIDTH),
                transforms.RandomHorizontalFlip(p=config.FLIP),
                transforms.ColorJitter(
                    brightness =config.BRIGHTNESS, 
                    contrast =config.CONTRAST, 
                    hue =config.HUE, 
                    saturation =config.SATURATION
                    ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[config.NORMALIZE_MEAN], std=[config.NORMALIZE_STD]),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(config.RESIZE_IMAGE),
                transforms.CenterCrop(config.HEIGHT_WIDTH),
                transforms.ToTensor(),
                transforms.Normalize(mean=[config.NORMALIZE_MEAN], std=[config.NORMALIZE_STD])
            ])

        test_transform = transforms.Compose([
            transforms.Resize(config.RESIZE_IMAGE),
            transforms.CenterCrop(config.HEIGHT_WIDTH),
            transforms.ToTensor(),
            transforms.Normalize(mean=[config.NORMALIZE_MEAN], std=[config.NORMALIZE_STD])
        ])

    return train_transform, test_transform




# ---------------------------------- DATA & DATASET CLASS ---------------------------------------


def load_data(data_name: str = "alzheimer"):
    """
    Load Alzheimer MRI or Skin Lesion dataset from Hugging Face Datasets.
    
    Args:
        data_name (str): The name of the dataset to load. Options are 'alzheimer' or 'skin lesions'.

    Returns:
        DatasetDict: A Hugging Face dataset.

    Raises:
        ValueError: If an unsupported dataset name is provided.
    """

    SUPPORTED_DATASETS = ["alzheimer", "skin_lesions"]

    if data_name.lower() not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset '{data_name}'. Available options are: {', '.join(SUPPORTED_DATASETS)}."
        )

    try:
        if data_name == "alzheimer":
            dataset = load_dataset('Falah/Alzheimer_MRI')
        else:  # data_name == 'skin_lesions'
            dataset = load_dataset("flwrlabs/fed-isic2019")

    except Exception:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        if data_name == "alzheimer":
            dataset = load_dataset('Falah/Alzheimer_MRI')
        else:
            dataset = load_dataset("flwrlabs/fed-isic2019")

    return dataset


class MedicalImageDataset(Dataset):
    def __init__(self, dataset_dict, partition_key="train", transform=None):
        self.partition = dataset_dict[partition_key]
        self.transform = transform

    def __getitem__(self, index):
        image = self.partition[index]["image"].convert("RGB")
        label = self.partition[index]["label"]
        
        # Preprocess image
        if self.transform:
            if isinstance(self.transform, A.BasicTransform) or isinstance(self.transform, A.Compose):
                # Albumentations transform
                image = np.array(image)
                image = self.transform(image=image)["image"]
            else:
                # Torchvision transform
                image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.partition)


# ------------------------------ LIGHTNING WRAPPER & DATAMODULE ---------------------------------------

class LightningModel(L.LightningModule):
    def __init__(self, model, learning_rate, base, num_labels):
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.base = base
        self.num_labels = num_labels
        self.dp_optimizer = None

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_labels)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_labels)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_labels)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        

    def forward(self, x):
        if self.base.lower() == "cnn":
            feed_forward = self.model(x)
        else:
            # if base is ViT
            feed_forward = self.model(x).logits
            
        return feed_forward

    def _shared_steps(self, batch):
        features, labels = batch

        logits = self(features)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)

        return loss, labels, preds

    def training_step(self, batch, batch_idx):
        loss, labels, preds = self._shared_steps(batch)

        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)

        self.train_acc(preds, labels)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, labels, preds = self._shared_steps(batch)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.val_acc(preds, labels)
        self.log("val_acc", self.val_acc, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        loss, labels, preds = self._shared_steps(batch)

        self.log("test_loss", loss, prog_bar=True)
        self.test_acc(preds, labels)
        self.log("accuracy", self.test_acc, prog_bar=True)

    def configure_optimizers(self):
        if self.dp_optimizer is not None:
            return self.dp_optimizer  #<--- Use Opacus-wrapped optimizer if set
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class MedicalImageDataModule(L.LightningDataModule):
    def __init__(self, data_name, batch_size, height_width, num_workers=NUM_WORKERS, augment_data=False):
        super().__init__()
        self.data_name = data_name
        self.batch_size = batch_size
        self.height_width = height_width
        self.num_workers = num_workers
        self.augment_data = augment_data
        self.dataset_dict = load_data(data_name)

        self.train_transform, self.test_transform = get_transforms(
            data_name=self.data_name,
            height_width=self.height_width,
            augment=self.augment_data
        )

    def setup(self, stage=None):
        self.train_dataset = MedicalImageDataset(dataset_dict=self.dataset_dict, partition_key="train", transform=self.train_transform)
        val_test_dataset = MedicalImageDataset(dataset_dict=self.dataset_dict, partition_key="test", transform=self.test_transform)
        self.val_dataset, self.test_dataset = random_split(dataset=val_test_dataset, lengths=[0.4, 0.6], generator=torch.Generator().manual_seed(SEED))

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )


#-------------------------------- LR FINDER & MODEL ARCHITECTURE HELPER FUNCTION -----------------------



def find_learning_rate(max_epoch: int, data_name: str, base_type: str, exp_name: ExperimentName, num_labels: int) -> float:
    """
    Runs a learning rate finder to suggest an optimal learning rate for training.

    This function sets up the datamodule and model based on the given dataset and base model type,
    then uses PyTorch Lightning's Tuner to search for a suitable learning rate. The learning rate
    curve is plotted and saved as a PDF.

    Args:
        max_epoch (int): Maximum number of epochs for the search.
        data_name (str): Name of the dataset.
        base_type (str): Base model architecture type ("cnn", "vit", etc.).
        exp_name (ExperimentName): Enum specifying the experiment configuration.

    Returns:
        float: Suggested optimal learning rate.
    """
    
    from lightning.pytorch.tuner import Tuner
    from lightning.pytorch.loggers import CSVLogger
    
    _config = load_yaml_config(key="experiments", item_name = exp_name)

    dm = MedicalImageDataModule(
        data_name = data_name,
        batch_size= _config["batch_size"],
        height_width = config.HEIGHT_WIDTH,
        num_workers= config.NUM_WORKERS,
        augment_data= config.AUGMENT)
    
    
    model = build_model(base_type, num_labels)

    lightning_model = LightningModel(model, learning_rate=_config.get("learning_rate"), base=base_type) 
    
    trainer = L.Trainer(accelerator="gpu", 
                        devices="auto", 
                        max_epochs=max_epoch, 
                        logger=CSVLogger(save_dir="../../logs/lr_finder", name="model"), 
                        deterministic=True)


    tuner = Tuner(trainer) #<--- create a tuner

    # find learning rate automatically
    lr_finder = tuner.lr_find(lightning_model, datamodule=dm)
    
    fig = lr_finder.plot(suggest=True)
    fig.savefig("../../logs/lr_finder/lr_suggest.pdf") #<--- save image
    
    new_lr = lr_finder.suggestion() #<--- Get suggestion
    
    return f"LR Suggestion: {new_lr}"


def build_model(base_type: str, num_labels: int) -> nn.Module:
    """
    Instantiates and returns the appropriate model architecture (CNN or ViT) based on the specified base type.

    Args:
        base_type (str): The type of model architecture to build. 
            - "cnn" returns a ResNet18 model.
            - "vit" returns a ViTForImageClassification model.

    Returns:
        nn.Module: The initialized PyTorch model moved to the available computing device (GPU or CPU).

    Example:
        >>> model = build_model(base_type="cnn")
        >>> print(model)
    """
    from transformers import ViTForImageClassification
    from torchvision.models import resnet18, ResNet18_Weights
    DEVICE = set_device()

    if base_type.lower() == "vit":
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
    else:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_labels)
    
    model.to(DEVICE)
    return model


# ---------------------------------- PLOTTING HELPER FUNCTIONS ---------------------------------------


def plot_csv_logger(csv_path, model_name, loss_names=["train_loss", "val_loss"], eval_names=["train_acc", "val_acc"]):
    metrics_data = pd.read_csv(csv_path)
    aggreg_metrics = []
    agg_col = "epoch"

    for i, dfg in metrics_data.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
        aggreg_metrics.append(agg)

    df_metrics = pd.DataFrame(aggreg_metrics)

    df_metrics[loss_names].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="Loss")
    plt.title("Loss vs. Epoch")
    plt.savefig(f"../artifacts/{model_name}--loss.pdf")

    df_metrics[eval_names].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="Accuracy")
    plt.title("Accuracy vs. Epoch")
    plt.savefig(f"../artifacts/{model_name}--acc.pdf")

    plt.show()


def predict_and_plot(model, trainer: L.Trainer, datamodule, num_labels, model_name: str = "", experiment_name: str =""):
    """
    Evaluates the trained model on a test dataset and computes classification metrics.

    Args:
        model (LightningModel): The trained model.
        trainer (L.Trainer): Lightning trainer instance.
        datamodule (MedicalImageDataModule): The datamodule containing test data.
        num_labels (int): Number of class labels in the data
        model_name (str): Trained model name
        experiment_name (str): The name of the experiment setup to run

    Returns:
        dict: Dictionary with evaluation metrics.
    """

    # Test model on test data
    model_result = trainer.test(
        model=model, datamodule=datamodule, ckpt_path="best")
    print(model_result)

    # Extract test accuracy and loss
    test_acc = model_result[0].get("accuracy", None)
    test_loss = model_result[0].get("test_loss", None)

    # Compute Additional Metrics
    y_true, y_pred, y_probs = [], [], []

    test_loader = datamodule.test_dataloader()
    device = set_device()

    model.to(device)
    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            probas = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probas.cpu().numpy())

    # Convert y_true, y_pred, and y_probs to PyTorch tensors
    y_true_tensor = torch.tensor(y_true, dtype=torch.long)
    y_pred_tensor = torch.tensor(y_pred, dtype=torch.long)
    y_probs_tensor = torch.tensor(np.array(y_probs), dtype=torch.float32)

    # Compute metrics using torchmetrics
    accuracy = Accuracy(task="multiclass", num_classes=num_labels)(
        y_pred_tensor, y_true_tensor)
    precision = Precision(task="multiclass", num_classes=num_labels,
                          average="weighted")(y_pred_tensor, y_true_tensor)
    recall = Recall(task="multiclass", num_classes=num_labels,
                    average="weighted")(y_pred_tensor, y_true_tensor)
    f1 = F1Score(task="multiclass", num_classes=num_labels,
                 average="weighted")(y_pred_tensor, y_true_tensor)
    mcc = MatthewsCorrCoef(task="multiclass", num_classes=num_labels)(
        y_pred_tensor, y_true_tensor)

    # Compute ROC and AUC
    roc = ROC(task="multiclass", num_classes=num_labels)
    auroc = AUROC(task="multiclass", num_classes=num_labels,
                  average="weighted")
    fpr, tpr, thresholds = roc(y_probs_tensor, y_true_tensor)
    roc_auc = auroc(y_probs_tensor, y_true_tensor)

    # Compute Confusion Matrix
    cmat = ConfusionMatrix(task="multiclass", num_classes=num_labels)(
        y_pred_tensor, y_true_tensor).numpy()

    # Print Metrics
    print(f"{model_name} Model Final Evaluation")
    print(
        f"Test Accuracy: {accuracy:.2%} | Precision: {precision:.2f} | Recall: {recall:.2f} | "
        f"F1-Score: {f1:.2f} | ROC-AUC: {roc_auc:.2f} | MCC: {mcc:.2f}"
    )

    # Plot Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    cmap = sns.color_palette('Blues_r')

    sns.heatmap(cmat, annot=True, cbar=False, cmap=cmap,
                ax=axes[0], annot_kws={"size": 15, "color": 'black'})
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('Actual Label')
    axes[0].set_title('Confusion Matrix', fontsize=15)

    # Plot ROC Curve
    for i in range(4):  # Assuming 4 classes
        axes[1].plot(fpr[i], tpr[i], lw=2, marker='.',label=f"Class {i} (AUC = {roc_auc:.2f})")
    axes[1].plot([0, 1], [0, 1], 'k--', label='Random Guess')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve', fontsize=15)
    axes[1].annotate(f'AUC ={round(roc_auc.item(), 2)}', xy=(0.7, 0.5), fontsize=15,)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # Print Classification Report
    print("\n", "_________" * 11)
    print(f"{model_name} Model Classification Report")
    print(classification_report(y_true, y_pred))
    print("_________" * 11)

    # Plot learning curve (assuming plot_csv_logger is defined)
    plot_csv_logger(f"{trainer.logger.log_dir}/metrics.csv", model_name=model_name)

    # Return final metrics
    final_model_result = {
        'Model': model_name,
        'Experiment Name': experiment_name,
        'Accuracy': round(accuracy.item(), 2),
        'Precision': round(precision.item(), 2),
        'Recall': round(recall.item(), 2),
        'F1-Score': round(f1.item(), 2),
        'ROC-AUC': round(roc_auc.item(), 2),
        'MCC': round(mcc.item(), 2)
    }
    
    # Log model performance to W&B
    wandb.log(final_model_result)
    
    return final_model_result


# ---------------------------------- FedML HELPER FUNCTIONS ---------------------------------------


def get_federated_config(name: str) -> dict:
    return load_yaml_config(key="experiments", item_name=name)


def get_weights(model) -> List[np.ndarray]:
    """
    Retrieves model parameters (weights & bias) from local model (client side).
    This function extracts the model's parameters as numpy arrays.

    Returns:
        List[np.ndarray]
            A list of numpy arrays representing the model's parameters. 
            Each numpy array in the list corresponds to parameters of a different layer or component of 
            the model.

    Examples:
        >>> model = YourModelClass()
        >>> parameters = get_parameters(model)
        >>> type(parameters)
        <class 'list'>
        >>> type(parameters[0])
        <class 'numpy.ndarray'>
    """
    weights = [w_params.cpu().numpy() for _, w_params in model.state_dict().items()]
    return weights


def set_weights(model, parameters):
    """
    Updates the model's parameters with new values provided as a list of NumPy ndarrays.

    This function takes a list of NumPy arrays containing new parameter values and updates the local model's
    parameters accordingly. It's typically used to set model parameters after they have been modified
    or updated elsewhere, possibly after aggregation in a federated learning scenario or after receiving
    updates from an optimization process.

    Parameters:
        parameters : List[np.ndarray]
            A list of NumPy ndarrays where each array corresponds to the parameters for a different layer or
            component of the model. The order of the arrays in the list should match the order of parameters
            in the model's state_dict.

    Returns:
        None

    Examples:
        >>> model = YourModelClass()
        >>> new_parameters = [np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([0.5, 0.6])]
        >>> set_parameters(model, new_parameters)
        >>> # Model parameters are now updated with `new_parameters`.

    Notes:
        - This method assumes that the provided list of parameters matches the structure and order of the model's parameters. If the order or structure of `parameters` does not match, this may lead to incorrect assignment of parameters or runtime errors.
        - The method converts each NumPy ndarray to a PyTorch tensor before updating the model's state dict. Ensure that the data types and device (CPU/GPU) of the NumPy arrays are compatible with your model's requirements.
    """
    params_layer_weight = zip(model.state_dict().keys(), parameters)
    params_state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_layer_weight})
    model.load_state_dict(params_state_dict, strict=True)


def simulate_clients(dataset: DatasetDict, num_clients: int, test_size: float = 0.08, seed: int = SEED) -> List[DatasetDict]:
    """
    Simulates federated learning clients from a given dataset by splitting the training data 
    among `num_clients`, with each client receiving its own train and validation set.

    Args:
        dataset (DatasetDict): A Hugging Face DatasetDict with "train" and "test" splits.
        test_size (float, optional): Proportion of each client's data to use for validation. Default is 0.08.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        List[DatasetDict]:
            - client_datasets: A list of DatasetDicts, one per client, each with:
                - "train": training data for that client
                - "val": validation data for that client
                - "test": the shared global test set (for evaluation)

    Raises:
        TypeError: If input is not a DatasetDict
        ValueError: If test_size is invalid or dataset missing required splits
    """

    # Input validation
    if not isinstance(dataset, DatasetDict):
        raise TypeError(
            "Input must be a DatasetDict with 'train' and 'test' keys")
    if not {'train', 'test'}.issubset(dataset.keys()):
        raise ValueError(
            "Dataset must contain both 'train', 'val', and 'test' splits")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    train_data = dataset["train"].shuffle(seed=seed)
    global_test_data = dataset['test']

    split_indices = np.array_split(range(len(train_data)), num_clients)

    client_datasets = []
    for indices in split_indices:
        client_subset = train_data.select(indices.tolist())
        client_split = client_subset.train_test_split(
            test_size=test_size, seed=seed)

        client_datasets.append(
            DatasetDict({
                "train": client_split["train"],
                "val": client_split['test'],
                "test": global_test_data
            })
        )
    return client_datasets


def _load_federated_config(exp_name: str, base_type: str) -> dict:
    """
    Loads the federated learning configuration for a given experiment and model type.

    This function maps the combination of an experiment name (e.g., 'fl', 'smpc', 'cdp-sf') and a 
    model type ('cnn' or 'vit') to the corresponding YAML configuration. It ensures that the 
    correct configuration is loaded for the specified experiment setup.

    Args:
        exp_name (str): The name of the federated experiment.
            Expected values include: "fl", "smpc", "cdp-sf", "cdp-sa", "cdp-cf", "cdp-ca", 
            "ldp-mod", "ldp-pe".
        base_type (str): The type of model architecture to use.
            Must be either "cnn" or "vit".

    Returns:
        dict: A dictionary containing the loaded federated learning configuration.

    Raises:
        ValueError: If an invalid combination of `exp_name` and `base_type` is provided.

    Example:
        >>> config = _load_federated_config(exp_name="fl", base_type="cnn")
        >>> print(config)
    """

    exp_name = exp_name.lower()
    base_type = base_type.lower()

    # all availabel valid combinations
    valid_config_dict = {
        ("fl", "cnn"): ExperimentName.FL_CNN,
        ("fl", "vit"): ExperimentName.FL_VIT,
        ("smpc", "cnn"): ExperimentName.FL_SMPC_CNN,
        ("smpc", "vit"): ExperimentName.FL_SMPC_VIT,
        ("cdp-sf", "cnn"): ExperimentName.FL_CDP_SF_CNN,
        ("cdp-sf", "vit"): ExperimentName.FL_CDP_SF_VIT,
        ("cdp-sa", "cnn"): ExperimentName.FL_CDP_SA_CNN,
        ("cdp-sa", "vit"): ExperimentName.FL_CDP_SA_VIT,
        ("cdp-cf", "cnn"): ExperimentName.FL_CDP_CF_CNN,
        ("cdp-cf", "vit"): ExperimentName.FL_CDP_CF_VIT,
        ("cdp-ca", "cnn"): ExperimentName.FL_CDP_CA_CNN,
        ("cdp-ca", "vit"): ExperimentName.FL_CDP_CA_VIT,
        ("ldp-mod", "cnn"): ExperimentName.FL_LDP_MOD_CNN,
        ("ldp-mod", "vit"): ExperimentName.FL_LDP_MOD_VIT,
        ("ldp-pe", "cnn"): ExperimentName.FL_LDP_PE_CNN,
        ("ldp-pe", "vit"): ExperimentName.FL_LDP_PE_VIT,
    }

    # Check if (exp_name, base_type) is valid
    key = (exp_name, base_type)
    if key not in valid_config_dict:
        valid_keys = ', '.join([f"{exp}/{base}" for exp, base in valid_config_dict.keys()])
        raise ValueError(
            f"Invalid combination: exp_name='{exp_name}', base_type='{base_type}'.\n"
            f"Valid options are: {valid_keys}."
        )

    return get_federated_config(valid_config_dict[key])



def prepare_FL_dataset(exp_name, data_name:str, base_type: Optional[Literal["cnn", "vit"]], augment_data: bool = config.AUGMENT):
    """
    Prepare federated learning datasets for multiple clients.

    This function loads the MedicalImage's dataset, applies specified data augmentation,
    partitions the data among clients, and returns DataLoaders for training, validation,
    and testing for each client.

    Args:
        exp_name: Name of the federated experiment setup to run
        data_name: Name of federated dataset to use in simulating the clients
        base_type (Optional[Literal["cnn", "vit"]]):
            The model architecture type to determine the preprocessing:
            - "cnn" for Convolutional Neural Networks
            - "vit" for Vision Transformers
        augment_data (bool, optional):
            Whether to apply data augmentation (random crop, horizontal flip, color jitter).
            Defaults to True.

    Returns:
        List[Tuple[DataLoader, DataLoader, DataLoader]]:
            A list where each element is a tuple containing the train, validation,
            and test DataLoaders for a client.
    """

    # Select federated config
    fed_config = _load_federated_config(exp_name=exp_name, base_type=base_type)

    num_clients = fed_config.get("num_clients")
    
    data = load_data(data_name)                                                                                                                                                  
    
    train_transform, test_transform = get_transforms(
            data_name=data_name,
            height_width=config.HEIGHT_WIDTH,
            augment=augment_data
        )

    
    clients_data_dict = simulate_clients(data, num_clients=num_clients, test_size=0.08)
    clients_dataloaders = []
    
    for i in range(len(clients_data_dict)):
        client_data = clients_data_dict[i]

        train_dataset = MedicalImageDataset(client_data,  partition_key="train", transform=train_transform)
        val_dataset = MedicalImageDataset(client_data,  partition_key="val", transform=train_transform)
        test_dataset = MedicalImageDataset(client_data,  partition_key="test", transform=test_transform)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=fed_config['batch_size'],
            pin_memory=True,
            num_workers=NUM_WORKERS,
            shuffle=True
        )

        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=fed_config['batch_size'],
            num_workers=NUM_WORKERS,
            shuffle=False
        )

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=fed_config['batch_size'],
            num_workers=NUM_WORKERS,
            shuffle=False
        )

        data_loaders = (train_loader, val_loader, test_loader)
        clients_dataloaders.append(data_loaders)

    return clients_dataloaders


def build_client_dataloaders(exp_name: str, base_type: str, data_name: str):
    """
    Prepares federated client dataloaders for a specific experiment configuration.

    Args:
        exp_name (str): The name of the experiment (e.g., "fl", "smpc", "cdp-sf").
        base_type (str): Model type ("cnn" or "vit") to determine preprocessing logic.
        data_name (str): Dataset name to load (e.g., "alzheimer", "skin_lesions").

    Returns:
        List[Tuple[DataLoader, DataLoader, DataLoader]]: 
            A list where each element contains (train_loader, val_loader, test_loader) for each simulated client.

    Example:
        >>> dataloaders = build_client_dataloaders(exp_name="fl", base_type="cnn", data_name="alzheimer")
    """
    return prepare_FL_dataset(
        exp_name=exp_name,
        base_type=base_type,
        data_name=data_name,
        augment_data=True,
    )
