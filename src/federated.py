import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time
import logging
import wandb
from tqdm import tqdm
from typing import Optional, Literal, Dict
from pathlib import Path
from codecarbon import EmissionsTracker
from opacus.validators import ModuleValidator

import torch
import torch.nn as nn

from sklearn.metrics import classification_report
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, ROC, MatthewsCorrCoef, ConfusionMatrix

from src.paths import EMISSIONS_LOG_DIR, DATA_DIR
from src.local_utility import set_device, set_seed, load_yaml_config
from src.tracker import _get_federated_emissions_summary, get_peak_memory_usage, start_memory_tracking, append_peak_memory, setup_wandb


DEVICE = set_device()

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


track_config = load_yaml_config(key="tracker")



#=================================================================
#                        TRAINING CODE 
# =================================================================


def train_model(model, train_loader, val_loader, client_id, base_type, fed_config, output_dir, optimizer=None):
    set_seed(seed_torch=True)
    
    tracker = EmissionsTracker(
                 project_name=f"Client {client_id}",
                 output_dir = Path(output_dir),
                 output_file = track_config.get("client_output_file"),
                 allow_multiple_runs = track_config.get("allow_multiple_runs", True),
                 log_level= track_config.get("log_level"),
                 measure_power_secs = track_config.get("measure_power_secs"),
                 save_to_file= True
             )

    start_memory_tracking()
    tracker.start()
     
    criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(fed_config['learning_rate']))

    model.to(DEVICE)
    scaler = torch.GradScaler()

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

                with torch.autocast(device_type=DEVICE):
                    logits = model(features) if base_type == "cnn" else model(features).logits
                    loss = criterion(logits, labels)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

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
            
            # Early stopping check
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

        end_time = time.perf_counter()
          
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
    
    total_time = end_time - start_time
    tracker.stop()
    
    cpu_peak, gpu_peak = get_peak_memory_usage()
    append_peak_memory(
        name=f"Client {client_id}", 
        cpu_mem=cpu_peak, 
        gpu_mem=gpu_peak, 
        source="client", 
        file_path = track_config.get('base_mem_dir'),
        duration = total_time
        )
        
    print(f"\n🔎 Tracker: {fed_config['name']}")
    print(f"📁 Logs saved to: {output_dir}/{track_config.get('client_output_file')}")
    print(f"⏱️ Total training time: {total_time//60:.0f} minutes {total_time % 60:.0f} seconds")
    
    return model, stats


def evaluate_model(model, val_loader, base_type):
    """
    Evaluates the given model on the validation dataset.

    Args:
        model (torch.nn.Module): The trained neural network model to be evaluated.
        val_set (torch.utils.data.Dataset): The validation dataset.
        base_type (str): The base model to use for validation on validation set: "cnn" or "vit".

    Returns:
        Tuple[float, float]: A tuple containing:
            - avg_loss (float): The average loss over the validation dataset.
            - accuracy (float): The accuracy of the model on the validation dataset.
    """
    
        
    correct, total_example, total_loss = 0, 0, 0
    criterion = nn.CrossEntropyLoss()

    model.to(DEVICE)

    with torch.no_grad():
        model.eval()
        for features, labels in val_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            logits = model(features) if base_type.lower() == "cnn" else model(features).logits

            predictions = torch.argmax(logits, dim=1)
            correct += torch.sum(predictions == labels).item()
            total_example += len(labels)
            total_loss += criterion(logits, labels).item()

    accuracy = correct / total_example
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, accuracy


def final_test_evaluation(model, test_loader, fed_config, data_name, base_type, num_labels, model_name: str = ""):
    
    # Start server-side CodeCarbon tracker
    experiment_name = fed_config["name"].replace("+ ", "").replace(" ", "_").replace("(", "").replace(")", "")
    experiment_log_dir = Path(track_config.get("output_dir")) / data_name / experiment_name
    experiment_log_dir.mkdir(parents=True, exist_ok=True)

    # W&B Logging Setup
    setup_wandb(
        config=fed_config,
        experiment_name=experiment_name,
        dataset_name=data_name
        )
    
    
    # Compute Additional Metrics
    y_true, y_pred, y_probs = [], [], []

    model.to(DEVICE)
    model.eval()
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images) if base_type.lower() == "cnn" else model(images).logits
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
    accuracy = Accuracy(task="multiclass", num_classes=num_labels)(y_pred_tensor, y_true_tensor)
    precision = Precision(task="multiclass", num_classes=num_labels,average="weighted")(y_pred_tensor, y_true_tensor)
    recall = Recall(task="multiclass", num_classes=num_labels,average="weighted")(y_pred_tensor, y_true_tensor)
    f1 = F1Score(task="multiclass", num_classes=num_labels,average="weighted")(y_pred_tensor, y_true_tensor)
    mcc = MatthewsCorrCoef(task="multiclass", num_classes=num_labels)(y_pred_tensor, y_true_tensor)

    # Compute ROC and AUC
    roc = ROC(task="multiclass", num_classes=num_labels)
    auroc = AUROC(task="multiclass", num_classes=num_labels, average="weighted")
    fpr, tpr, thresholds = roc(y_probs_tensor, y_true_tensor)
    roc_auc = auroc(y_probs_tensor, y_true_tensor)

    # Compute Confusion Matrix
    cmat = ConfusionMatrix(task="multiclass", num_classes=num_labels)(y_pred_tensor, y_true_tensor).numpy()

    # Print Metrics
    print("\n", "_________" * 11)
    print(f"{model_name} Model Final Evaluation \n")
    print(
        f"Test Accuracy: {accuracy:.2%} | Precision: {precision:.2f} | Recall: {recall:.2f} | "
        f"F1-Score: {f1:.2f} | ROC-AUC: {roc_auc:.2f} | MCC: {mcc:.2f}"
    )
    print(" ")

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
        axes[1].plot(fpr[i], tpr[i], lw=2, marker='.',
                     label=f"Class {i} (AUC = {roc_auc:.2f})")
    axes[1].plot([0, 1], [0, 1], 'k--', label='Random Guess')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve', fontsize=15)
    axes[1].annotate(f'AUC ={round(roc_auc.item(), 2)}', xy=(0.7, 0.5), fontsize=15,)
    axes[1].legend()

    plt.show()

    # Print Classification Report
    print("\n", "_________" * 11)
    print(f"{model_name}--{data_name} Model Classification Report")
    print(classification_report(y_true, y_pred))
    print("_________" * 11)

    
    # print the energy metrics

    _ = _get_federated_emissions_summary(experiment_log_dir, fed_config)

    
    # Return final model metrics
    final_model_metrics = {
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
    wandb.log(final_model_metrics)
    wandb.finish()
    
    return final_model_metrics
