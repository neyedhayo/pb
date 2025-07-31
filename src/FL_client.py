from pathlib import Path
from typing import Dict
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from flwr.client import NumPyClient
from flwr.common import Scalar, NDArrays

from src.federated import train_model, evaluate_model
from src.privacy_engine import train_FL_model
from src.local_utility import set_device, set_weights, get_weights, load_yaml_config, _load_federated_config


DEVICE = set_device()
track_config = load_yaml_config(key="tracker")


class MedicalImageClient(NumPyClient):
    def __init__(self, model, train_loader, val_loader, exp_name: str, data_name: str, base_type: str, client_id: int, optim=None, use_privacy_engine:bool = False, target_delta = None, noise_multiplier = None, max_grad_norm = None, output_dir: Path = Path(track_config.get('output_dir'))):
        super().__init__()
        self.device = DEVICE
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.exp_name = exp_name.lower()
        self.data_name = data_name
        self.base_type = base_type.lower()
        self.client_id = client_id
        self.optimizer = optim
        self.output_dir = output_dir
        self.privacy_engine = use_privacy_engine
        self.target_delta = float(target_delta) if self.privacy_engine else target_delta
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm

        # Select federated config
        self.fed_config = _load_federated_config(self.exp_name, self.base_type)

        # Setup experiment directory
        experiment_name = self.fed_config["name"].replace(" + ", "_").replace(" ", "_").replace("(", "").replace(")", "")
        self.experiment_log_dir = self.output_dir / self.data_name/ experiment_name
        self.experiment_log_dir.mkdir(parents=True, exist_ok=True)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]):
        set_weights(self.model, parameters)
        
        if not self.privacy_engine:
            train_model(
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                client_id=self.client_id,
                base_type=self.base_type,
                fed_config=self.fed_config,
                output_dir=self.experiment_log_dir,
                optimizer=self.optimizer
                )
            
        else:
            # Use Opacus Privacy Engine
            if any(x is None for x in [self.target_delta, self.noise_multiplier, self.max_grad_norm]):
                raise ValueError(
                    f"[Client {self.client_id}] Privacy Engine is enabled but one or more required parameters are missing:\n"
                    f"target_delta={self.target_delta}, "
                    f"noise_multiplier={self.noise_multiplier}, "
                    f"max_grad_norm={self.max_grad_norm}\n"
                    f"Please ensure all are set when use_privacy_engine=True."
                )
                
            _, _, epsilon =  train_FL_model(
                model = self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                client_id=self.client_id,
                base_type=self.base_type,
                fed_config=self.fed_config,
                target_delta = self.target_delta,
                noise_multiplier= self.noise_multiplier,
                max_grad_norm = self.max_grad_norm,
                output_dir=self.experiment_log_dir,
                optimizer=self.optimizer 
            )
            
            if epsilon is not None:
                print(f"Epsilon value for delta={self.target_delta} is {epsilon:.2f}")
            else:
                print("Epsilon value not available.")

        return get_weights(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        set_weights(self.model, parameters)
        loss, accuracy = evaluate_model(self.model, self.val_loader, self.base_type)
        return loss, len(self.val_loader.dataset), {"accuracy": accuracy}
    
    