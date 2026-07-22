import random

import numpy as np
import torch

from .config import *
from .dataset import HuaweiForecastDataset
from .model import HuaweiForecastTransformer
from .inference import format_prediction_output, print_prediction_output


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

# ==========================================================
# Evaluate
# ==========================================================

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

            sample["past_target"].unsqueeze(0).to(DEVICE),

            sample["function"].unsqueeze(0).to(DEVICE),

            sample["region"].unsqueeze(0).to(DEVICE),

            sample["cluster"].unsqueeze(0).to(DEVICE),

            sample["category"].unsqueeze(0).to(DEVICE),

            sample["stability"].unsqueeze(0).to(DEVICE)

        )

        result = format_prediction_output(

            prediction["mu"].cpu().numpy()[0],

            prediction["sigma"].cpu().numpy()[0],

            sample["past_target"].numpy()

        )

        print_prediction_output(result)
