import torch

from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Subset

from .config import *

from .dataset import HuaweiForecastDataset
from .model import HuaweiForecastTransformer
from .loss import get_loss
from .trainer import Trainer


# ==========================================================
# Load Dataset
# ==========================================================

print()

print("=" * 70)
print("Loading Dataset")
print("=" * 70)

dataset = HuaweiForecastDataset()

print()

print(f"Total Sequences : {len(dataset):,}")

# -------------------------------------------------
# Optional development subset
# -------------------------------------------------

SUBSET_SIZE = None

if SUBSET_SIZE is not None:

    dataset = Subset(
        dataset,
        range(SUBSET_SIZE)
    )

    print(f"Training on {SUBSET_SIZE:,} samples")

# ----------------------------------------------------------
# Development mode
# ----------------------------------------------------------
#
# Uncomment during debugging.
#
# from torch.utils.data import Subset
# dataset = Subset(dataset, range(50000))
#
# ----------------------------------------------------------

train_size = int(

    TRAIN_SPLIT *

    len(dataset)

)

validation_size = (

    len(dataset)

    -

    train_size

)

train_dataset, validation_dataset = random_split(

    dataset,

    [

        train_size,

        validation_size

    ],

    generator=torch.Generator().manual_seed(

        RANDOM_SEED

    )

)

print()

print(f"Training Samples   : {len(train_dataset):,}")

print(f"Validation Samples : {len(validation_dataset):,}")

# ==========================================================
# DataLoaders
# ==========================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY,

    persistent_workers=PERSISTENT_WORKERS

)

validation_loader = DataLoader(

    validation_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY,

    persistent_workers=PERSISTENT_WORKERS

)

# ==========================================================
# Build Model
# ==========================================================

model = HuaweiForecastTransformer(

    num_functions=dataset.dataset.num_functions,

    num_regions=dataset.dataset.num_regions,

    num_clusters=dataset.dataset.num_clusters,

    num_categories=dataset.dataset.num_categories,

    num_stability=dataset.dataset.num_stability

)

model = model.to(

    DEVICE

)

print()

print("=" * 70)

print("Model")

print("=" * 70)

print()

print(model)

# ==========================================================
# Optimizer
# ==========================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.5,

    patience=2

)

criterion = get_loss()

# ==========================================================
# Trainer
# ==========================================================

trainer = Trainer(

    model=model,

    optimizer=optimizer,

    criterion=criterion,

    scheduler=scheduler,

    device=DEVICE

)

# ==========================================================
# Train
# ==========================================================

trainer.fit(

    train_loader,

    validation_loader,

    EPOCHS

)

print()

print("=" * 70)

print("Training Complete")

print("=" * 70)
