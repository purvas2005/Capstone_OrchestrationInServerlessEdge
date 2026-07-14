import numpy as np
import torch

from .config import *
from .dataset import HuaweiForecastDataset
from .model import HuaweiForecastTransformer


# ==========================================================
# Inference Engine
# ==========================================================

class ForecastEngine:

    def __init__(self):

        self.dataset = HuaweiForecastDataset()

        self.model = HuaweiForecastTransformer(

            num_functions=self.dataset.num_functions,

            num_regions=self.dataset.num_regions,

            num_clusters=self.dataset.num_clusters,

            num_categories=self.dataset.num_categories,

            num_stability=self.dataset.num_stability

        ).to(DEVICE)

        checkpoint = torch.load(

            MODEL_DIR / CHECKPOINT_NAME,

            map_location=DEVICE

        )

        if "model_state_dict" in checkpoint:

            self.model.load_state_dict(

                checkpoint["model_state_dict"]

            )

        else:

            self.model.load_state_dict(

                checkpoint

            )

        self.model.eval()

    # ------------------------------------------------------

    @torch.no_grad()

    def predict(self, sample):

        prediction = self.model(

            sample["past_values"].unsqueeze(0).to(DEVICE),

            sample["past_time_features"].unsqueeze(0).to(DEVICE),

            sample["future_time_features"].unsqueeze(0).to(DEVICE),

            sample["function"].unsqueeze(0).to(DEVICE),

            sample["region"].unsqueeze(0).to(DEVICE),

            sample["cluster"].unsqueeze(0).to(DEVICE),

            sample["category"].unsqueeze(0).to(DEVICE),

            sample["stability"].unsqueeze(0).to(DEVICE)

        )

        mu = prediction["mu"].cpu().numpy()[0]

        sigma = prediction["sigma"].cpu().numpy()[0]

        # ------------------------------------------
        # Undo log transform
        # ------------------------------------------

        mu = np.expm1(mu)

        sigma = np.expm1(sigma)

        lower = np.maximum(

            0,

            mu - 1.96 * sigma

        )

        upper = mu + 1.96 * sigma

        return {

            "forecast": mu,

            "uncertainty": sigma,

            "lower95": lower,

            "upper95": upper

        }

    # ------------------------------------------------------

    def predict_dataset_index(

        self,

        index

    ):

        sample = self.dataset[index]

        result = self.predict(sample)

        result["ground_truth"] = np.expm1(

            sample["target"].numpy()

        )

        return result


# ==========================================================
# Example
# ==========================================================

if __name__ == "__main__":

    engine = ForecastEngine()

    result = engine.predict_dataset_index(0)

    print()

    print("=" * 70)

    print("Forecast")

    print("=" * 70)

    print()

    print("Prediction")

    print(np.round(result["forecast"], 2))

    print()

    print("Ground Truth")

    print(np.round(result["ground_truth"], 2))

    print()

    print("95% Lower")

    print(np.round(result["lower95"], 2))

    print()

    print("95% Upper")

    print(np.round(result["upper95"], 2))
