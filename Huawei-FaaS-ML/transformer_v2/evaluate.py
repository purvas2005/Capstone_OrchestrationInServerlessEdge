"""Evaluate the trained forecaster on a chronological holdout.

This intentionally does not alter ``test.py`` or the user-facing forecast
format.  It gives one reproducible accuracy number for model selection and
compares the model to a last-value persistence baseline.
"""

import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .config import BATCH_SIZE, CHECKPOINT_NAME, DEVICE, MODEL_DIR, NUM_WORKERS, PIN_MEMORY
from .dataset import HuaweiForecastDataset
from .model import HuaweiForecastTransformer


@torch.no_grad()
def main():
    dataset = HuaweiForecastDataset()
    _, validation_indices = dataset.temporal_split_indices()
    validation = Subset(dataset, validation_indices)
    loader = DataLoader(
        validation,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    model = HuaweiForecastTransformer(
        num_functions=dataset.num_functions,
        num_regions=dataset.num_regions,
        num_clusters=dataset.num_clusters,
        num_categories=dataset.num_categories,
        num_stability=dataset.num_stability,
    ).to(DEVICE)
    checkpoint = torch.load(MODEL_DIR / CHECKPOINT_NAME, map_location=DEVICE)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    model_errors, baseline_errors = [], []
    for batch in loader:
        inputs = {
            name: batch[name].to(DEVICE, non_blocking=True)
            for name in (
                "past_values", "past_time_features", "future_time_features",
                "past_target", "function", "region", "cluster", "category", "stability",
            )
        }
        target_log = batch["target"].to(DEVICE, non_blocking=True)
        prediction = model(**inputs)

        # The model is Gaussian in log1p space, so this is its request-space
        # expectation, not simply exp(mu)-1.
        request_prediction = torch.expm1(prediction["mu"] + 0.5 * prediction["sigma"].square()).clamp_min(0)
        request_target = torch.expm1(target_log).clamp_min(0)
        persistence = torch.expm1(inputs["past_target"][:, -1:]).expand_as(request_target)

        model_errors.append((request_prediction - request_target).detach().cpu().numpy().ravel())
        baseline_errors.append((persistence - request_target).detach().cpu().numpy().ravel())

    model_errors = np.concatenate(model_errors)
    baseline_errors = np.concatenate(baseline_errors)

    def metrics(errors):
        return float(np.mean(np.abs(errors))), float(math.sqrt(np.mean(errors ** 2)))

    model_mae, model_rmse = metrics(model_errors)
    base_mae, base_rmse = metrics(baseline_errors)

    print("=" * 70)
    print("Chronological Holdout Accuracy (request count space)")
    print("=" * 70)
    print(f"Samples             : {len(validation):,}")
    print(f"Model MAE           : {model_mae:.4f}")
    print(f"Model RMSE          : {model_rmse:.4f}")
    print(f"Persistence MAE     : {base_mae:.4f}")
    print(f"Persistence RMSE    : {base_rmse:.4f}")
    print(f"MAE improvement     : {(1 - model_mae / base_mae) * 100:.2f}%")


if __name__ == "__main__":
    main()
