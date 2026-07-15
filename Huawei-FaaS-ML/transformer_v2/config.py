from pathlib import Path
import torch

# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATABASE_DIR = PROJECT_ROOT / "database"

MODEL_DIR = PROJECT_ROOT / "models"

LOG_DIR = PROJECT_ROOT / "logs"

DB_PATH = DATABASE_DIR / "huawei.duckdb"

DATABASE_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ==========================================================
# Dataset
# ==========================================================

FEATURE_TABLE = "feature_table"

SEQUENCE_LENGTH = 60

PREDICTION_HORIZON = 10

TRAIN_SPLIT = 0.90

VALID_SPLIT = 0.10

RANDOM_SEED = 42

# ==========================================================
# Feature Columns
# ==========================================================

PAST_VALUE_FEATURES = [

    "requests",

    "avg_cpu",
    "avg_memory",
    "avg_runtime",
    "avg_request_size",

    "active_pods",
    "active_users",

    "lag_1",
    "lag_5",
    "lag_15",
    "lag_30",

    "rolling_mean_5",
    "rolling_mean_15",
    "rolling_mean_30",

    "rolling_std",

    "rolling_max",
    "rolling_min",
    "rolling_median",

    "warm_capacity",

    "trend",
    "growth_rate",

    "coeff_variation",

    "cpu_per_request",
    "memory_per_request",
    "runtime_per_request",

    "warm_ratio"

]

TIME_FEATURES = [

    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos"

]

TARGET_COLUMN = "requests"

# ==========================================================
# Transformer
# ==========================================================

D_MODEL = 128

NHEAD = 8

NUM_ENCODER_LAYERS = 4

NUM_DECODER_LAYERS = 4

DIM_FEEDFORWARD = 512

DROPOUT = 0.10

# ==========================================================
# Embedding Dimensions
# ==========================================================

FUNCTION_EMBED_DIM = 64

REGION_EMBED_DIM = 8

CLUSTER_EMBED_DIM = 8

CATEGORY_EMBED_DIM = 8

STABILITY_EMBED_DIM = 8

STATIC_EMBED_DIM = (

    FUNCTION_EMBED_DIM +

    REGION_EMBED_DIM +

    CLUSTER_EMBED_DIM +

    CATEGORY_EMBED_DIM +

    STABILITY_EMBED_DIM

)

# ==========================================================
# Training
# ==========================================================

BATCH_SIZE = 64

EPOCHS = 20

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

GRADIENT_CLIP = 1.0

EARLY_STOPPING_PATIENCE = 5

NUM_WORKERS = 4

PIN_MEMORY = torch.cuda.is_available()

PERSISTENT_WORKERS = NUM_WORKERS > 0

# ==========================================================
# Loss
# ==========================================================

LOSS_FUNCTION = "gaussian"
# Options:
# "mse"
# "gaussian"

# ==========================================================
# Device
# ==========================================================

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)

# ==========================================================
# Logging
# ==========================================================

PRINT_EVERY = 50

CHECKPOINT_NAME = "forecast_transformer.pt"

TRAIN_LOG = MODEL_DIR / "training_log.csv"
