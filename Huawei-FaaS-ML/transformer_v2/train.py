import torch
import numpy as np

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

train_dataset, validation_dataset, test_dataset = dataset.temporal_split()


def deterministic_subsample(source, maximum, seed):

    """Keep a reproducible uniform subset without contaminating the holdout."""

    if maximum is None or len(source) <= maximum:
        return source

    generator = torch.Generator().manual_seed(seed)
    selected = torch.randperm(len(source), generator=generator)[:maximum].tolist()
    return Subset(source, selected)


def balanced_activity_subsample(source, maximum, seed, active_fraction):
    """Sample training windows evenly by whether their future horizon has demand."""

    if maximum is None or len(source) <= maximum:
        return source

    active, inactive = [], []
    for local_index in range(len(source)):
        source_index = source.source_index(local_index)
        bucket = active if source.dataset.window_has_activity(source_index) else inactive
        bucket.append(local_index)

    rng = np.random.default_rng(seed)
    desired_active = min(int(round(maximum * active_fraction)), len(active))
    desired_inactive = min(maximum - desired_active, len(inactive))

    # If one bucket is smaller than its target, fill the remainder from the
    # other bucket rather than duplicating windows.
    remaining = maximum - desired_active - desired_inactive
    if remaining:
        extra_active = min(remaining, len(active) - desired_active)
        desired_active += extra_active
        desired_inactive += min(remaining - extra_active, len(inactive) - desired_inactive)

    selected = np.concatenate((
        rng.choice(active, size=desired_active, replace=False),
        rng.choice(inactive, size=desired_inactive, replace=False),
    ))
    rng.shuffle(selected)
    print(
        "Balanced training sample: "
        f"{desired_active:,} active-horizon / {desired_inactive:,} inactive-horizon windows"
    )
    return Subset(source, selected.tolist())


train_dataset = balanced_activity_subsample(

    train_dataset,

    PILOT_TRAIN_SAMPLES,

    RANDOM_SEED,

    ACTIVE_TRAIN_FRACTION,

)

validation_dataset = deterministic_subsample(

    validation_dataset,

    PILOT_VALIDATION_SAMPLES,

    RANDOM_SEED + 1

)

print()

print(f"Training Samples   : {len(train_dataset):,}")

print(f"Validation Samples : {len(validation_dataset):,}")

print(f"Test Samples       : {len(test_dataset):,} (held out)")

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
