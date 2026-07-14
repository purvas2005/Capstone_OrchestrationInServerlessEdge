import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from .config import *
from .dataset import HuaweiForecastDataset
from .model import HuaweiForecastTransformer


# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 70)
print("Loading Dataset")
print("=" * 70)

dataset = HuaweiForecastDataset()

# ==========================================================
# Build Model
# ==========================================================

model = HuaweiForecastTransformer(

    num_functions=dataset.num_functions,

    num_regions=dataset.num_regions,

    num_clusters=dataset.num_clusters,

    num_categories=dataset.num_categories,

    num_stability=dataset.num_stability

).to(DEVICE)

# ==========================================================
# Load Checkpoint
# ==========================================================

checkpoint = torch.load(

    MODEL_DIR / CHECKPOINT_NAME,

    map_location=DEVICE

)

if "model_state_dict" in checkpoint:

    model.load_state_dict(

        checkpoint["model_state_dict"]

    )

else:

    model.load_state_dict(

        checkpoint

    )

model.eval()

print()

print("Checkpoint Loaded")

# ==========================================================
# Evaluate
# ==========================================================

mae_scores = []

rmse_scores = []

mape_scores = []

r2_scores = []

NUM_SAMPLES = 20

with torch.no_grad():

    for sample_idx in range(NUM_SAMPLES):

        index = random.randint(

            0,

            len(dataset)-1

        )

        sample = dataset[index]

        prediction = model(

            sample["past_values"].unsqueeze(0).to(DEVICE),

            sample["past_time_features"].unsqueeze(0).to(DEVICE),

            sample["future_time_features"].unsqueeze(0).to(DEVICE),

            sample["function"].unsqueeze(0).to(DEVICE),

            sample["region"].unsqueeze(0).to(DEVICE),

            sample["cluster"].unsqueeze(0).to(DEVICE),

            sample["category"].unsqueeze(0).to(DEVICE),

            sample["stability"].unsqueeze(0).to(DEVICE)

        )

        prediction = prediction["mu"].cpu().numpy()[0]

        target = sample["target"].numpy()

        # ----------------------------------------------
        # Undo log transform
        # ----------------------------------------------

        prediction = np.expm1(prediction)

        target = np.expm1(target)

        mae = mean_absolute_error(

            target,

            prediction

        )

        rmse = np.sqrt(

            mean_squared_error(

                target,

                prediction

            )

        )

        r2 = r2_score(

            target,

            prediction

        )

        mape = np.mean(

            np.abs(

                (target - prediction)

                /

                np.maximum(target,1)

            )

        ) * 100

        mae_scores.append(mae)

        rmse_scores.append(rmse)

        r2_scores.append(r2)

        mape_scores.append(mape)

        print()

        print("="*60)

        print(f"Sample {sample_idx+1}")

        print("="*60)

        print()

        print("Prediction")

        print(

            np.round(

                prediction,

                2

            )

        )

        print()

        print("Ground Truth")

        print(

            np.round(

                target,

                2

            )

        )

        plt.figure(

            figsize=(8,4)

        )

        plt.plot(

            target,

            marker="o",

            linewidth=2,

            label="Actual"

        )

        plt.plot(

            prediction,

            marker="x",

            linewidth=2,

            label="Prediction"

        )

        plt.xlabel(

            "Future Minute"

        )

        plt.ylabel(

            "Requests"

        )

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        plt.show()

# ==========================================================
# Overall Metrics
# ==========================================================

print()

print("="*70)

print("Evaluation Summary")

print("="*70)

print()

print(

    f"MAE   : {np.mean(mae_scores):.3f}"

)

print(

    f"RMSE  : {np.mean(rmse_scores):.3f}"

)

print(

    f"MAPE  : {np.mean(mape_scores):.2f}%"

)

print(

    f"R²    : {np.mean(r2_scores):.3f}"

)
