import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from transformers import ViTForImageClassification

import logging
import time
from functools import wraps
from pathlib import Path
from typing import List, Tuple
from codecarbon import EmissionsTracker

from flwr.common import Metrics
from flwr.server import ServerApp, LegacyContext, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
from flwr.common import ndarrays_to_parameters
from flwr.common.logger import update_console_handler

from src import config 
from src.local_utility import (
    set_device, 
    set_weights,
    get_weights,
    build_client_dataloaders, 
    build_model,
    load_yaml_config,
    _load_federated_config,
)
from src.federated import evaluate_model, final_test_evaluation
from src.tracker import start_memory_tracking, append_peak_memory, get_peak_memory_usage


DEVICE = set_device()


#------------------------------- TRACK SERVER-SIDE EMISSION ------------------------------

def track_server_emissions(experiment_item: str, data_name : str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            server_round = args[0]
            fed_config = load_yaml_config(key="experiments", item_name=experiment_item)
            track_config = load_yaml_config(key="tracker")

            experiment_name = (
                fed_config["name"]
                .replace("+ ", "")
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
            )
            output_dir = Path(track_config.get("output_dir")) / data_name / experiment_name
            output_dir.mkdir(parents=True, exist_ok=True)

            tracker = EmissionsTracker(
                project_name=f"Server Round {server_round}",
                output_dir=str(output_dir),
                output_file=track_config.get("server_output_file"),
                allow_multiple_runs=track_config.get("allow_multiple_runs"),
                measure_power_secs=track_config.get("measure_power_secs"),
                save_to_file=True,
                log_level=track_config.get("log_level"),
            )
            
            
            start_memory_tracking()
            tracker.start()
            start_time = time.perf_counter()
            loss, accuracy = func(*args, **kwargs)
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            tracker.stop()
            
            cpu_peak, gpu_peak = get_peak_memory_usage()
            append_peak_memory(
                name=f"Server Round {server_round}",
                cpu_mem=cpu_peak,
                gpu_mem=gpu_peak,
                source="server",
                file_path=track_config.get("base_mem_dir"),
                duration= total_time
            )

            return loss, accuracy

        return wrapper

    return decorator


# ---------------------------------- FLOWER SERVER -----------------------------------


def build_server_app(evaluate_fn, metrics_fn, fed_config, base_type: str, num_labels: int) -> ServerApp:
    """
    Builds a Flower ServerApp for SMPC with dynamic model and evaluation.
    """

    app = ServerApp()

    @app.main()
    def main(grid, context):
        # Select model
        if base_type == "vit":
            model = ViTForImageClassification.from_pretrained(
                pretrained_model_name_or_path="google/vit-base-patch16-224-in21k",
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
            )
        else:
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, num_labels)

        model.to(DEVICE)
        initial_params = ndarrays_to_parameters(get_weights(model))

        # Global Server Model Update Aggregation Strategy
        strategy = FedAvg(
            fraction_fit=1.0,                   # <--- Sample 100% of available clients for training
            fraction_evaluate=1.0,              # <--- Sample 100% of available clients for evaluation
            initial_parameters=initial_params,  # <--- Initial model parameters
            evaluate_fn=evaluate_fn,            # <--- Global evaluation function
            evaluate_metrics_aggregation_fn=metrics_fn, # <-- pass the metric aggregation function
        )

        context = LegacyContext(
            context=context,
            config=ServerConfig(num_rounds=fed_config.get("num_rounds")),  # <--- no. of federated rounds
            strategy=strategy,
        )

        update_console_handler("DEBUG")

        # Create fit workflow
        # For further information, please see:
        # https://flower.ai/docs/framework/ref-api/flwr.server.workflow.SecAggPlusWorkflow.html
        workflow = DefaultWorkflow(
            fit_workflow=SecAggPlusWorkflow(
                num_shares=fed_config.get("num_shares"),
                reconstruction_threshold=fed_config.get("reconstruction_threshold"),
                max_weight=fed_config.get("max_weight", 2000),
                clipping_range= fed_config.get("clipping_range", 8.0),
                quantization_range= fed_config.get("quantization_range"),
                modulus_range= fed_config.get("modulus_range"),
                timeout= fed_config.get("timeout", 30.0)
            )
        )

        workflow(grid, context)

    return app



# ========================= GLOBAL SERVER EVALUATION FUNCTIONS ===============================


def build_evaluate_fn(exp_name: str, base_type: str, data_name: str, experiment_item: str, num_labels: str):
    """
    Builds a global evaluation function for the server, customized to the experiment setup.

    This function wraps the evaluation logic, applying emissions tracking, 
    dynamic model loading, dataset preparation, weight setting, and evaluation reporting.

    Args:
        exp_name (str): The name of the experiment (e.g., "fl", "smpc", "cdp-sf").
        base_type (str): Model type ("cnn" or "vit") indicating the architecture to build.
        data_name (str): Dataset name for loading evaluation data (e.g., "alzheimer", "skin_lesions").
        experiment_item (str): The experiment name key used to fetch YAML configuration and set emissions tracking.

    Returns:
        Callable: A decorated evaluation function ready to be passed to the Flower server strategy.

    Example:
        >>> evaluate_fn = build_evaluate_fn(
        >>>     exp_name="cdf-sf",
        >>>     base_type="vit",
        >>>     data_name="skin_lesions",
        >>>     experiment_item=ExperimentName.FL_SMPC_VIT
        >>> )
    """

    @track_server_emissions(experiment_item=experiment_item, data_name=data_name)
    def evaluate_fn(server_round, parameters, config):
        client_dataloaders = build_client_dataloaders(
            exp_name=exp_name,
            base_type=base_type,
            data_name=data_name,
        )
        fed_config = _load_federated_config(exp_name, base_type)

        model = build_model(base_type=base_type, num_labels=num_labels)
        set_weights(model, parameters)

        _, _, test_loader = client_dataloaders[0]

        loss, accuracy = evaluate_model(model, test_loader, base_type=base_type)

        # Run detailed evaluation only at the last round
        if server_round == fed_config.get("num_rounds"):
            # Wait briefly to ensure all emission logs are flushed
            time.sleep(1)
            final_test_evaluation(
                model,
                test_loader,
                fed_config,
                data_name,
                base_type=base_type,
                num_labels=num_labels,
                model_name=experiment_item,
            )

        return loss, {"accuracy": accuracy}

    return evaluate_fn



# ================================ EVAL METRICS AGGREGATION FUNCTION ===================================


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Compute the weighted average accuracy across multiple clients.

    This function calculates the weighted average of accuracy scores
    from multiple clients, where the weight is the number of examples
    each client contributes.

    Args:
        metrics (List[Tuple[int, Metrics]]):
            A list of tuples, where each tuple contains:
            - num_examples (int): The number of examples used by a client.
            - m (Metrics): A dictionary containing accuracy metrics, e.g., {"accuracy": value}.

    Returns:
        Metrics:
            A dictionary containing the aggregated accuracy metric:
            - "accuracy" (float): The weighted average accuracy across all clients.

    Example:
        >>> metrics = [(100, {"accuracy": 0.85}), (200, {"accuracy": 0.90})]
        >>> weighted_average(metrics)
        {"accuracy": 0.8833}
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


