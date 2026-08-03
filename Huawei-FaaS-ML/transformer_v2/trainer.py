from pathlib import Path
import csv

import torch
from torch.nn.utils import clip_grad_norm_

from .config import *
from .distribution import hurdle_request_quantile


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
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

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

                    "validation_mae",

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
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.use_amp,
            ):

                prediction = self.model(

                    batch["past_values"].to(self.device, non_blocking=True),

                    batch["past_time_features"].to(self.device, non_blocking=True),

                    batch["future_time_features"].to(self.device, non_blocking=True),

                    batch["past_target"].to(self.device, non_blocking=True),

                    batch["function"].to(self.device, non_blocking=True),

                    batch["region"].to(self.device, non_blocking=True),

                    batch["cluster"].to(self.device, non_blocking=True),

                    batch["category"].to(self.device, non_blocking=True),

                    batch["stability"].to(self.device, non_blocking=True)

                )

                target = batch["target"].to(

                    self.device,

                    non_blocking=True

                )

                loss = self.criterion(

                    prediction,

                    target
                )

            if self.use_amp:

                self.scaler.scale(loss).backward()

                self.scaler.unscale_(self.optimizer)

                clip_grad_norm_(

                    self.model.parameters(),

                    GRADIENT_CLIP
                )

                self.scaler.step(self.optimizer)

                self.scaler.update()

            else:

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

                    f"Objective={loss.item():.5f}"

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

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.use_amp,
            ):

                prediction = self.model(

                    batch["past_values"].to(self.device, non_blocking=True),

                    batch["past_time_features"].to(self.device, non_blocking=True),

                    batch["future_time_features"].to(self.device, non_blocking=True),

                    batch["past_target"].to(self.device, non_blocking=True),

                    batch["function"].to(self.device, non_blocking=True),

                    batch["region"].to(self.device, non_blocking=True),

                    batch["cluster"].to(self.device, non_blocking=True),

                    batch["category"].to(self.device, non_blocking=True),

                    batch["stability"].to(self.device, non_blocking=True)

                )

                target = batch["target"].to(

                    self.device,

                    non_blocking=True
                )

                loss = self.criterion(

                    prediction,

                    target

                )

            validation_loss += loss.item()

        validation_loss /= len(loader)

        return validation_loss

    @torch.no_grad()

    def validate_request_mae(self, loader):

        """MAE of the zero-inflated distribution median in request space."""

        self.model.eval()
        total_error = 0.0
        total_values = 0

        for batch in loader:

            prediction = self.model(

                batch["past_values"].to(self.device, non_blocking=True),
                batch["past_time_features"].to(self.device, non_blocking=True),
                batch["future_time_features"].to(self.device, non_blocking=True),
                batch["past_target"].to(self.device, non_blocking=True),
                batch["function"].to(self.device, non_blocking=True),
                batch["region"].to(self.device, non_blocking=True),
                batch["cluster"].to(self.device, non_blocking=True),
                batch["category"].to(self.device, non_blocking=True),
                batch["stability"].to(self.device, non_blocking=True),

            )

            forecast = hurdle_request_quantile(
                prediction["mu"],
                prediction["sigma"],
                prediction["occurrence_probability"],
                0.5,
            )
            target = torch.expm1(batch["target"].to(self.device, non_blocking=True)).clamp_min(0.0)
            total_error += torch.abs(forecast - target).sum().item()
            total_values += target.numel()

        return total_error / total_values

    @torch.no_grad()

    def validate_persistence_mae(self, loader):

        """MAE of repeating the final observed request count."""

        total_error = 0.0
        total_values = 0

        for batch in loader:

            target = torch.expm1(
                batch["target"].to(self.device, non_blocking=True)
            ).clamp_min(0.0)
            persistence = torch.expm1(
                batch["past_target"].to(self.device, non_blocking=True)[:, -1:]
            ).expand_as(target)
            total_error += torch.abs(persistence - target).sum().item()
            total_values += target.numel()

        return total_error / total_values
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

            validation_mae = self.validate_request_mae(

                val_loader

            )

            persistence_mae = self.validate_persistence_mae(

                val_loader

            )

            if self.scheduler is not None:

                self.scheduler.step(

                validation_mae

                )

            current_lr = self.optimizer.param_groups[0]["lr"]

            print()

            print(f"Train Loss      : {train_loss:.6f}")

            print(f"Validation Loss : {validation_loss:.6f}")

            print(f"Validation MAE  : {validation_mae:.6f}")

            print(f"Persistence MAE : {persistence_mae:.6f}")

            print(

                f"MAE Improvement : "

                f"{(1.0 - validation_mae / persistence_mae) * 100:.2f}%"

            )

            print(f"Learning Rate   : {current_lr:.8f}")

            # ------------------------------------------
            # Save Best Model
            # ------------------------------------------

            if validation_mae < self.best_loss:

                self.best_loss = validation_mae

                self.wait = 0

                checkpoint = {

                    "epoch": epoch + 1,

                    "model_state_dict":
                        self.model.state_dict(),

                    "optimizer_state_dict":
                        self.optimizer.state_dict(),

                    "validation_loss":
                        validation_loss,

                    "validation_mae":
                        validation_mae

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

                    validation_mae,

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

            f"Best Validation MAE : "

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
