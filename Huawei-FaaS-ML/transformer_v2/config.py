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

WARM_CAPACITY_LOOKBACK = 30

# Chronological partitions within each function series.  The final test tail
# remains untouched while model choices are made using validation only.
TRAIN_SPLIT = 0.80

VALID_SPLIT = 0.10

TEST_SPLIT = 0.10

RANDOM_SEED = 42

# Irreducible uncertainty in log1p(requests) space.  This prevents Gaussian
# NLL from collapsing predicted variance on repeated low-load sequences.
MIN_LOG_SIGMA = 0.05

# The lognormal mean grows exponentially with sigma.  Bounding sigma prevents
# a few pathological uncertainty estimates from becoming enormous request
# forecasts and dominating request-space error.
MAX_LOG_SIGMA = 1.00

# The direct request-count MAE is scaled to keep it numerically comparable to
# Gaussian NLL while still selecting a useful central forecast.
REQUEST_MAE_LOSS_WEIGHT = 1.0
REQUEST_MAE_LOSS_SCALE = 1000.0

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

# Dense minute series yield roughly ten times as many windows as the sparse
# event-only table.  Validate the new data semantics with this bounded run
# before intentionally removing these limits for final training.
# CPU-friendly pilot.  It validates the dense-data pipeline before a longer
# experiment on a GPU-capable instance.
PILOT_TRAIN_SAMPLES = 25_000
PILOT_VALIDATION_SAMPLES = 5_000
PILOT_EPOCHS = 5
PILOT_EVALUATION_SAMPLES = 5_000
# Loading all 6,949 dense function series exceeds the RAM of small training
# instances.  Keep complete series for a deterministic pilot subset so its
# chronological validation remains valid.
PILOT_MAX_FUNCTION_GROUPS = 300

# Random windows overwhelmingly contain no requests in the dense trace.
# Balance only the training subset by whether its future horizon has activity;
# validation and test data retain their natural distributions.
ACTIVE_TRAIN_FRACTION = 0.50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

GRADIENT_CLIP = 1.0

EARLY_STOPPING_PATIENCE = 5

NUM_WORKERS = 2

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

# Pilot checkpoints are isolated from the promoted full model so experiments
# can be compared without overwriting the currently serving checkpoint.
CHECKPOINT_NAME = (
    "forecast_transformer_pilot.pt"
    if PILOT_TRAIN_SAMPLES is not None
    else "forecast_transformer.pt"
)

TRAIN_LOG = (
    MODEL_DIR / "training_log_pilot.csv"
    if PILOT_TRAIN_SAMPLES is not None
    else MODEL_DIR / "training_log_full.csv"
)
