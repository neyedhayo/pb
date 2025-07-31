import os
from pathlib import Path

PARENT_DIR = Path(__file__).parent.resolve().parent

DATA_DIR = PARENT_DIR / "data"
CLIENTS_DATA_DIR = DATA_DIR / "clients_data"
SEED_DATA_DIR = DATA_DIR/"seed_data"

LOGS_DIR = PARENT_DIR / "logs"
EMISSIONS_LOG_DIR = LOGS_DIR / "emissions"
RAY_LOG_DIR = LOGS_DIR/"ray"

SOURCE_DIR = PARENT_DIR / "src"
MODELS_DIR = PARENT_DIR / "models"


if not Path(DATA_DIR).exists():
    os.makedirs(DATA_DIR, exist_ok=True)
    
if not Path(SEED_DATA_DIR).exists():
    os.makedirs(SEED_DATA_DIR, exist_ok=True)

if not Path(MODELS_DIR).exists():
    os.makedirs(MODELS_DIR, exist_ok=True)

if not Path(LOGS_DIR).exists():
    os.makedirs(LOGS_DIR, exist_ok=True)

if not Path(EMISSIONS_LOG_DIR).exists():
    os.makedirs(EMISSIONS_LOG_DIR, exist_ok=True)

if not Path(RAY_LOG_DIR).exists():
    os.makedirs(RAY_LOG_DIR, exist_ok=True)

if not Path(SOURCE_DIR).exists():
    os.makedirs(SOURCE_DIR, exist_ok=True)
