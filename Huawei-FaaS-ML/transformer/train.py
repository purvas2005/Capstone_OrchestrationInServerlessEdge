from pathlib import Path
import csv
import torch
from torch.utils.data import (
    DataLoader,
    random_split,
    Subset
)

from .dataset import HuaweiSequenceDataset
from .model import TransformerPredictor
from .trainer import train_one_epoch, evaluate
from .config import *

# ============================================================
# Configuration
# ============================================================

USE_SUBSET = True
SUBSET_SIZE = 100000

CHECKPOINT_DIR = Path("models")
CHECKPOINT_DIR.mkdir(exist_ok=True)

LOG_FILE = CHECKPOINT_DIR / "training_log.csv"

# ============================================================
# Load Dataset
# ============================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

dataset = HuaweiSequenceDataset()

if USE_SUBSET:

    subset_size = min(SUBSET_SIZE, len(dataset))

    dataset = Subset(
        dataset,
        range(subset_size)
    )

print(f"Dataset Size : {len(dataset):,}")

# ============================================================
# Train / Validation Split
# ============================================================

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

print(f"Training Samples   : {len(train_dataset):,}")
print(f"Validation Samples : {len(val_dataset):,}")

# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# ============================================================
# Build Model
# ============================================================

sample = dataset[0]
input_size = sample["x"].shape[1]

print(f"Input Features : {input_size}")

model = TransformerPredictor(
    input_size=input_size,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    prediction_horizon=PREDICTION_HORIZON
).to(DEVICE)

criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)

# ============================================================
# CSV Logger
# ============================================================

if not LOG_FILE.exists():

    with open(LOG_FILE, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "learning_rate"
        ])

# ============================================================
# Training
# ============================================================

print()
print("=" * 70)
print("TRAINING")
print("=" * 70)

best_val_loss = float("inf")

for epoch in range(EPOCHS):

    print()

    print(f"Epoch {epoch+1}/{EPOCHS}")

    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        DEVICE
    )

    val_loss = evaluate(
        model,
        val_loader,
        criterion,
        DEVICE
    )

    scheduler.step(val_loss)

    lr = optimizer.param_groups[0]["lr"]

    print(f"Train Loss : {train_loss:.6f}")
    print(f"Validation : {val_loss:.6f}")
    print(f"Learning Rate : {lr:.8f}")

    # --------------------------------------
    # Save Epoch Checkpoint
    # --------------------------------------

    epoch_path = CHECKPOINT_DIR / f"epoch_{epoch+1}.pt"

    torch.save(
        model.state_dict(),
        epoch_path
    )

    # --------------------------------------
    # Save Best Model
    # --------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            CHECKPOINT_DIR / "best_model.pt"
        )

        print("✓ Best model updated")

    # --------------------------------------
    # Append CSV Log
    # --------------------------------------

    with open(LOG_FILE, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            epoch + 1,
            train_loss,
            val_loss,
            lr
        ])

print()

print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(f"Best Validation Loss : {best_val_loss:.6f}")
print(f"Models saved in      : {CHECKPOINT_DIR}")
print(f"Training log         : {LOG_FILE}")
