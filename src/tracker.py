import os
import torch
import time
import datetime
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import wandb
import tracemalloc
from codecarbon import EmissionsTracker

from src.paths import EMISSIONS_LOG_DIR, SOURCE_DIR, DATA_DIR, LOGS_DIR
from src.local_utility import load_yaml_config
from src.config import WANDB_PROJECT, WANDB_ENTITY



# ---------------------------------- EXPERIMENT TRACKING ---------------------------------------



def setup_wandb(config: dict, experiment_name: str, dataset_name: str):
    """
    Initialize Weight-&-Bias logging.
    Args:
        config (dict): Hyperparameters and experiment config.
        experiment_name (str): Name of the experiment.
        dataset_name (str): Name of the dataset used.
    """
    
    load_dotenv()
    
    try:
        if wandb.run is None:
            wandb.login(key=os.getenv("WANDB_API_KEY"))
            wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=f"{experiment_name}_{dataset_name}",
                dir = LOGS_DIR,
                config={
                    **config,
                    "dataset": dataset_name
                }
            )
    except wandb.errors.UsageError:
        print("[WARNING] W&B login failed. Switching to offline mode.")
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(
            project="PrivacyBench",
            name=experiment_name.value,
            dir = LOGS_DIR,
            config={
                **config,
                "dataset": dataset_name
            }
        )



# -------------------------------- MEMORY, ENERGY, AND EMISSION TRACKING -----------------------------------


def reset_base_memory_csv():
    """
    Resets (overwrites) the base_memory.csv file for peak memory tracking.
    """
    file = DATA_DIR / "base_memory.csv"
    empty_df = pd.DataFrame(columns=["name", "cpu_peak", "gpu_peak", "source", "timestamp"])
    empty_df.to_csv(file, index=False)


def append_peak_memory(name: str, cpu_mem: float, gpu_mem: float, source: str, file_path: Path, duration: float):
    
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"{file_path} not found")
    
    df = pd.read_csv(file_path)
    new_row = {
        "name": name,
        "cpu_peak": cpu_mem,
        "gpu_peak": gpu_mem,
        "source": source,
        "time_duration": duration,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(str(file_path), index=False)  


def get_peak_memory_usage(stop_tracking=True, reset_gpu=True):
    """
    Get peak memory usage
    
    Args:
        stop_tracking: Whether to stop tracemalloc after measurement
        reset_gpu: Whether to reset CUDA stats after measurement
    Returns:
        Tuple of (cpu_peak_gb, gpu_peak_gb)
    """
    cpu_peak = tracemalloc.get_traced_memory()[1] / (1024 ** 3) if tracemalloc.is_tracing() else 0
    
    gpu_peak = 0
    if torch.cuda.is_available():
        gpu_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        if reset_gpu:
            torch.cuda.reset_peak_memory_stats()
    
    if stop_tracking and tracemalloc.is_tracing():
        tracemalloc.stop()
    
    return cpu_peak, gpu_peak


def start_memory_tracking():
    """
    Start fresh memory tracking session
    """
    if tracemalloc.is_tracing():
        tracemalloc.stop()
    tracemalloc.start()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _get_baseline_emissions_summary(output_dir: Path) -> pd.Series:

    track_config = load_yaml_config(key="tracker")
    emissions_file = output_dir / track_config.get("output_file")

    if not emissions_file.exists():
        raise FileNotFoundError(f"No emissions file found at: {emissions_file}")

    df = pd.read_csv(emissions_file)
    
    # Append memory info to the latest emissions entry
    cpu_gb, gpu_gb = get_peak_memory_usage()
    df["peak_cpu_gb"] = cpu_gb
    df["peak_gpu_gb"] = gpu_gb
    df.to_csv(emissions_file, index=False)
    
    metrics = df.iloc[-1].to_dict()

    print("\n📊 From emissions.csv:")
    print(f"⏱️ Total Duration: {metrics['duration']:.2f} sec")
    print(f"✅ Total Energy consumed: {metrics['energy_consumed']:.5f} kWh")
    print(f"🌍 Total CO₂ emitted: {metrics['emissions']:.5f} kg")
    print(f"🧠 Peak CPU RAM: {metrics['peak_cpu_gb']:.2f} GB")
    print(f"🖥️ Peak GPU VRAM: {metrics['peak_gpu_gb']:.2f} GB")
    
    # Log to W&B
    wandb.log({
    "Total Training Duration (sec)": metrics['duration'],
    "Energy Consumed (kWh)": metrics['energy_consumed'],
    "CO2 Emitted (kg)": metrics['emissions'],
    "Peak CPU RAM (GB)": metrics['peak_cpu_gb'],
    "Peak GPU VRAM (GB)": metrics['peak_gpu_gb']
    })
    
    return metrics


def track_emissions(experiment_name: str, data_name: str, yaml_path: str = f"{SOURCE_DIR}/experiments.yaml", base_log_dir: Path = EMISSIONS_LOG_DIR):
    """
    Decorator factory to track energy consumption and CO₂ emissions of training runs using CodeCarbon.

    This function returns a decorator that wraps around a training function to:
      - Load experiment-specific hyperparameters from a YAML file.
      - Configure and start a CodeCarbon EmissionsTracker.
      - Measure training time.
      - Save emissions data to a structured directory.
      - Print a summary of duration, energy used, and emissions.

    The wrapped function is expected to return a Lightning `DataModule`, `Trainer`, and model.

    Args:
        experiment_name (str): Name of the experiment to match in the YAML config (e.g., "CNN Baseline").
        data_name (str): Name of the dataset the model trained on.
        yaml_path (str, optional): Path to the YAML file containing experiment and tracker configurations.
                                   Defaults to "src/experiments.yaml".
        base_log_dir (Path, optional): Directory where emissions data should be saved.
                                       A subfolder will be created using the experiment name.

    Returns:
        Callable: A decorator that wraps a training function and tracks its emissions.

    Raises:
        FileNotFoundError: If the YAML file or emissions.csv is not found.
        ValueError: If the specified experiment or config section is not found in the YAML file.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            exp_config = load_yaml_config(yaml_path, key="experiments", item_name=experiment_name)
            track_config = load_yaml_config(yaml_path, key="tracker")
            output_dir = base_log_dir / data_name/ experiment_name.replace(" ", "_")
            output_dir.mkdir(parents=True, exist_ok=True)

            tracker = EmissionsTracker(
                project_name=experiment_name,
                output_dir=str(output_dir),
                output_file=track_config.get("output_file"),
                allow_multiple_runs=track_config.get("allow_multiple_runs", False),
                log_level=track_config.get("log_level"),
                measure_power_secs=track_config.get("measure_power_secs")
            )

            tracker.start()
            start_time = time.perf_counter()
            
            # Initialize W&B Logging
            setup_wandb(config=exp_config, experiment_name=experiment_name, dataset_name=data_name)

            # Experiment Config gets injected here
            dm, trainer, lightning_model = func(*args, **kwargs, config=exp_config)

            end_time = time.perf_counter()
            total_time = end_time - start_time
            tracker.stop()

            print(f"\n🔎 Tracker: {experiment_name}")
            print(
                f"📁 Logs saved to: {output_dir}/{track_config.get('output_file')}")
            print(f"⏱️ Total training time: {total_time:.2f} seconds")
            
            
            _ = _get_baseline_emissions_summary(output_dir)
            
            return dm, trainer, lightning_model
        return wrapper
    return decorator


def _get_federated_emissions_summary(output_dir: Path, config: Dict) -> pd.DataFrame:
    track_config = load_yaml_config(key="tracker")
    server_log_file = output_dir / track_config.get("server_output_file")
    client_log_file = output_dir / track_config.get("client_output_file")

    expected_client_rows = config.get("num_rounds") * config.get("num_clients")
    expected_server_rows = config.get("num_rounds")

    dfs = []
    
    
    # Process server logs
    if server_log_file.exists():

        df_server = pd.read_csv(server_log_file).tail(expected_server_rows)
        df_server['source'] = 'server'
        df_server.to_csv(server_log_file, index=False)
        
        dfs.append(df_server)
    else:
        raise FileNotFoundError(f"Cannot find {server_log_file}")
    
    # Process client logs
    if client_log_file.exists():
        df_client = pd.read_csv(client_log_file).tail(expected_client_rows)
        df_client['source'] = 'client'
        dfs.append(df_client)

    if not dfs:
        raise FileNotFoundError("No emissions files found for server or client!")

    full_df = pd.concat(dfs, ignore_index=True)
    full_df.to_csv(output_dir / "consolidated_emissions.csv", index=False)
    
    if not (DATA_DIR / "base_memory.csv").exists():
        raise FileNotFoundError(f'{DATA_DIR / "base_memory.csv"} not found')
    
    memory_df = pd.read_csv(f"{DATA_DIR}/base_memory.csv")
    
    #assert len(memory_df) == len(full_df), "base_memory.csv not updated completely"
    full_df = pd.concat([full_df, memory_df], axis=1)
    full_df.to_csv(output_dir / "consolidated_emissions.csv", index=False)
    


    # Calculate totals
    total_duration = full_df["time_duration"].sum()
    total_energy = full_df['energy_consumed'].sum()
    total_emissions = full_df['emissions'].sum()
    peak_cpu = memory_df['cpu_peak'].max()
    peak_gpu = memory_df['gpu_peak'].max()

    print("\n📊 Consolidated Metrics:")
    print(f"⏱️ Total Duration: {total_duration:.2f} sec")
    print(f"✅ Total Energy: {total_energy:.5f} kWh")
    print(f"🌍 Total CO₂ Emitted: {total_emissions:.5f} kg")
    print(f"🧠 Peak CPU RAM: {peak_cpu:.4f} GB")
    print(f"🖥️ Peak GPU VRAM: {peak_gpu:.4f} GB")
    print("\n","_________" * 11)
    
    
    # Log consolidated federated metrics to W&B
    wandb.log({
        "Total FL Duration (sec)": total_duration,
        "Total FL Energy (kWh)": total_energy,
        "Total FL CO2 Emitted (kg)": total_emissions,
        "Peak FL CPU RAM (GB)": peak_cpu,
        "Peak FL GPU VRAM (GB)": peak_gpu,
        })
    
    return full_df
    
