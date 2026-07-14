import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from .dataset import HuaweiSequenceDataset
from .model import TransformerPredictor
from .config import *

# -----------------------
# Load Dataset
# -----------------------

dataset = HuaweiSequenceDataset()

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True
)

sample = dataset[0]
input_size = sample["x"].shape[1]

# -----------------------
# Load Model
# -----------------------

model = TransformerPredictor(
    input_size=input_size,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    prediction_horizon=PREDICTION_HORIZON
).to(DEVICE)

model.load_state_dict(
    torch.load(
        "models/transformer_v1.pt",
        map_location=DEVICE
    )
)

model.eval()

# -----------------------
# Test 10 Random Samples
# -----------------------

with torch.no_grad():

    for i, batch in enumerate(loader):

        if i == 10:
            break

        x = batch["x"].to(DEVICE)

        prediction = model(x)

        prediction = prediction.cpu().numpy()[0]
	prediction = np.expm1(prediction.cpu().numpy()[0])

        target = np.expm1(batch["y"].numpy()[0])

        print("="*60)
        print(f"Sample {i+1}")

        print("Prediction")

        print(np.round(prediction,2))

        print()

        print("Ground Truth")

        print(np.round(target,2))

        print()

        plt.figure(figsize=(7,4))

        plt.plot(
            target,
            label="Actual",
            marker="o"
        )

        plt.plot(
            prediction,
            label="Prediction",
            marker="x"
        )

        plt.xlabel("Future Minute")

        plt.ylabel("Requests")

        plt.title(f"Forecast {i+1}")

        plt.grid(True)

        plt.legend()

        plt.show()
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

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

print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")
