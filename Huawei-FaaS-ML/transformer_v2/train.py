import torch

from torch.utils.data import DataLoader
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

train_indices, validation_indices = dataset.temporal_split_indices()


def deterministic_subsample(indices, maximum, seed):

    """Keep a reproducible uniform subset without contaminating the holdout."""

    if maximum is None or len(indices) <= maximum:
        return indices

    generator = torch.Generator().manual_seed(seed)
    selected = torch.randperm(len(indices), generator=generator)[:maximum].tolist()
    return [indices[index] for index in selected]


train_indices = deterministic_subsample(

    train_indices,

    PILOT_TRAIN_SAMPLES,

    RANDOM_SEED

)

validation_indices = deterministic_subsample(

    validation_indices,

    PILOT_VALIDATION_SAMPLES,

    RANDOM_SEED + 1

)

train_dataset = Subset(dataset, train_indices)

validation_dataset = Subset(dataset, validation_indices)

print()

print(f"Training Samples   : {len(train_dataset):,}")

print(f"Validation Samples : {len(validation_dataset):,}")

if PILOT_TRAIN_SAMPLES is not None:

    print("Pilot mode: deterministic training/validation subsets are active.")

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

    num_functions=dataset.num_functions,

    num_regions=dataset.num_regions,

    num_clusters=dataset.num_clusters,

    num_categories=dataset.num_categories,

    num_stability=dataset.num_stability

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

    PILOT_EPOCHS if PILOT_TRAIN_SAMPLES is not None else EPOCHS

)

print()

print("=" * 70)

print("Training Complete")

print("=" * 70)
