from pathlib import Path
import csv

import torch
from torch.nn.utils import clip_grad_norm_

from .config import *


# ==========================================================
# Trainer
# ==========================================================

class Trainer:

    def __init__(

        self,

        model,

        optimizer,

        criterion,

        scheduler,

        device=DEVICE

    ):

        self.model = model

        self.optimizer = optimizer

        self.criterion = criterion

        self.scheduler = scheduler

        self.device = device

        self.best_loss = float("inf")

        self.wait = 0

        self.model_dir = Path(MODEL_DIR)

        self.model_dir.mkdir(

            exist_ok=True,

            parents=True

        )

        self.log_path = TRAIN_LOG

        if not self.log_path.exists():

            with open(

                self.log_path,

                "w",

                newline=""

            ) as f:

                writer = csv.writer(f)

                writer.writerow([

                    "epoch",

                    "train_loss",

                    "validation_loss",

                    "learning_rate"

                ])

    # ======================================================

    def train_epoch(

        self,

        loader

    ):

        self.model.train()

        running_loss = 0.0

        for batch_idx, batch in enumerate(loader):

            self.optimizer.zero_grad()

            prediction = self.model(

                batch["past_values"].to(self.device),

                batch["past_time_features"].to(self.device),

                batch["future_time_features"].to(self.device),

                batch["function"].to(self.device),

                batch["region"].to(self.device),

                batch["cluster"].to(self.device),

                batch["category"].to(self.device),

                batch["stability"].to(self.device)

            )

            target = batch["target"].to(

                self.device

            )

            loss = self.criterion(

                prediction,

                target

            )

            loss.backward()

            clip_grad_norm_(

                self.model.parameters(),

                GRADIENT_CLIP

            )

            self.optimizer.step()

            running_loss += loss.item()

            if batch_idx % PRINT_EVERY == 0:

                print(

                    f"Batch "

                    f"{batch_idx}/{len(loader)} "

                    f"Loss={loss.item():.5f}"

                )

        running_loss /= len(loader)

        return running_loss

    # ======================================================

    @torch.no_grad()

    def validate(

        self,

        loader

    ):

        self.model.eval()

        validation_loss = 0.0

        for batch in loader:

            prediction = self.model(

                batch["past_values"].to(self.device),

                batch["past_time_features"].to(self.device),

                batch["future_time_features"].to(self.device),

                batch["function"].to(self.device),

                batch["region"].to(self.device),

                batch["cluster"].to(self.device),

                batch["category"].to(self.device),

                batch["stability"].to(self.device)

            )

            target = batch["target"].to(

                self.device

            )

            loss = self.criterion(

                prediction,

                target

            )

            validation_loss += loss.item()

        validation_loss /= len(loader)

        return validation_loss
    # ======================================================
    # Complete Training
    # ======================================================

    def fit(

        self,

        train_loader,

        val_loader,

        epochs

    ):

        print()

        print("=" * 70)

        print("Starting Training")

        print("=" * 70)

        for epoch in range(epochs):

            print()

            print(f"Epoch {epoch+1}/{epochs}")

            print("-" * 70)

            train_loss = self.train_epoch(

                train_loader

            )

            validation_loss = self.validate(

                val_loader

            )

            if self.scheduler is not None:

                self.scheduler.step(

                    validation_loss

                )

            current_lr = self.optimizer.param_groups[0]["lr"]

            print()

            print(f"Train Loss      : {train_loss:.6f}")

            print(f"Validation Loss : {validation_loss:.6f}")

            print(f"Learning Rate   : {current_lr:.8f}")

            # ------------------------------------------
            # Save Best Model
            # ------------------------------------------

            if validation_loss < self.best_loss:

                self.best_loss = validation_loss

                self.wait = 0

                checkpoint = {

                    "epoch": epoch + 1,

                    "model_state_dict":
                        self.model.state_dict(),

                    "optimizer_state_dict":
                        self.optimizer.state_dict(),

                    "validation_loss":
                        validation_loss

                }

                torch.save(

                    checkpoint,

                    self.model_dir /

                    CHECKPOINT_NAME

                )

                print()

                print("✓ Best model updated")

            else:

                self.wait += 1

                print()

                print(

                    f"No improvement "

                    f"({self.wait}/"

                    f"{EARLY_STOPPING_PATIENCE})"

                )

            # ------------------------------------------
            # CSV Logging
            # ------------------------------------------

            with open(

                self.log_path,

                "a",

                newline=""

            ) as f:

                writer = csv.writer(f)

                writer.writerow([

                    epoch + 1,

                    train_loss,

                    validation_loss,

                    current_lr

                ])

            # ------------------------------------------
            # Early Stopping
            # ------------------------------------------

            if self.wait >= EARLY_STOPPING_PATIENCE:

                print()

                print("=" * 70)

                print("Early Stopping")

                print("=" * 70)

                break

        print()

        print("=" * 70)

        print("Training Finished")

        print("=" * 70)

        print()

        print(

            f"Best Validation Loss : "

            f"{self.best_loss:.6f}"

        )

        print()

        print(

            f"Checkpoint saved to"

        )

        print(

            self.model_dir /

            CHECKPOINT_NAME

        )

        print()

        print(

            f"Training log saved to"

        )

        print(

            self.log_path

        )
