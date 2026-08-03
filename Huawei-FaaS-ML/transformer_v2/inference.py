import numpy as np
import torch

from .config import *
from .dataset import HuaweiForecastDataset
from .model import HuaweiForecastTransformer


# ==========================================================
# Forecast Output Formatting
# ==========================================================

Q10_Z = -1.281551565545
Q50_Z = 0.0
Q90_Z = 1.281551565545


def _lognormal_moments(mu, sigma):

    variance = (np.exp(sigma ** 2) - 1.0) * np.exp(2.0 * mu + sigma ** 2)

    mean = np.exp(mu + 0.5 * sigma ** 2) - 1.0

    std = np.sqrt(np.maximum(variance, 1e-12))

    return mean, std


def _lognormal_quantile(mu, sigma, z_score):

    return np.maximum(0.0, np.expm1(mu + z_score * sigma))


def _normal_cdf(x, mean, std):

    distribution = torch.distributions.Normal(

        torch.as_tensor(mean, dtype=torch.float32),

        torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6)

    )

    return distribution.cdf(

        torch.as_tensor(x, dtype=torch.float32)

    ).cpu().numpy()


def _warm_capacity_from_history(past_target):

    recent = past_target[-WARM_CAPACITY_LOOKBACK:]

    recent = recent[recent > 0]

    if recent.size == 0:

        return 0.0

    return float(np.percentile(recent, 90))


def _hurdle_moments(mu, sigma, probability):

    conditional_mean, _ = _lognormal_moments(mu, sigma)
    conditional_second_moment = (
        np.exp(2.0 * mu + 2.0 * sigma ** 2)
        - 2.0 * np.exp(mu + 0.5 * sigma ** 2)
        + 1.0
    )
    mean = probability * conditional_mean
    variance = probability * conditional_second_moment - mean ** 2
    return mean, np.sqrt(np.maximum(variance, 1e-12))


def _hurdle_quantile(mu, sigma, probability, quantile):

    probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
    conditional_quantile = (quantile - (1.0 - probability)) / probability
    positive = conditional_quantile > 0.0
    conditional_quantile = np.clip(conditional_quantile, 1e-6, 1.0 - 1e-6)
    normal = torch.distributions.Normal(0.0, 1.0)
    z_values = normal.icdf(torch.as_tensor(conditional_quantile, dtype=torch.float32)).numpy()
    values = np.maximum(0.0, np.expm1(mu + z_values * sigma))
    return np.where(positive, values, 0.0)


def format_prediction_output(mu, sigma, past_target, occurrence_probability=None):

    if occurrence_probability is None:
        occurrence_probability = np.ones_like(mu)

    mean, std = _hurdle_moments(mu, sigma, occurrence_probability)

    quantiles = {

        "q10": _hurdle_quantile(mu, sigma, occurrence_probability, 0.10),

        "q50": _hurdle_quantile(mu, sigma, occurrence_probability, 0.50),

        "q90": _hurdle_quantile(mu, sigma, occurrence_probability, 0.90),

    }

    warm_capacity = _warm_capacity_from_history(

        np.expm1(past_target)

    )

    # P(Y > capacity) for a hurdle distribution.  Positive demand follows a
    # lognormal count distribution; the zero atom never exceeds capacity.
    conditional_risk = 1.0 - _normal_cdf(
        np.log1p(warm_capacity), mu, sigma
    )
    cold_start_risk = occurrence_probability * conditional_risk

    return {

        "mean": mean,

        "std": std,
        "quantiles": quantiles,
        "post_processing": {

            "cold_start_risk": cold_start_risk,

            "warm_capacity": warm_capacity,

        }

    }


def print_prediction_output(result):

    print()

    print("Outputs of the transformer")

    print()

    print("Mean (μ)")

    print("- Expected number of requests in future time steps")

    print(np.round(result["mean"], 2))

    print()

    print("Standard Deviation (σ)")

    print("- Uncertainty in predicted workload")

    print(np.round(result["std"], 2))

    print()

    print("Quantile Estimates")

    print("- q10, q50 (median), q90")

    print("- Captures range of possible future demand")

    print("q10:", np.round(result["quantiles"]["q10"], 2))

    print("q50:", np.round(result["quantiles"]["q50"], 2))

    print("q90:", np.round(result["quantiles"]["q90"], 2))

    print()

    print("Post Processing Outputs:")

    print()

    print("Cold Start Risk:")

    print("- Probability that predicted demand exceeds capacity")

    print("- Calculated Using: P(X > capacity), where X ~ N(μ, σ)")

    print(np.round(result["post_processing"]["cold_start_risk"], 4))

    print()

    print("Warm Capacity")

    print("- Take recent history (across N timestamps)")

    print("- Ignore zeros")

    print("- Compute 90th percentile")

    print("- capacity = value below which 90% of recent loads fall")

    print(np.round(result["post_processing"]["warm_capacity"], 2))


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

            sample["past_target"].unsqueeze(0).to(DEVICE),

            sample["function"].unsqueeze(0).to(DEVICE),

            sample["region"].unsqueeze(0).to(DEVICE),

            sample["cluster"].unsqueeze(0).to(DEVICE),

            sample["category"].unsqueeze(0).to(DEVICE),

            sample["stability"].unsqueeze(0).to(DEVICE)

        )

        mu = prediction["mu"].cpu().numpy()[0]

        sigma = prediction["sigma"].cpu().numpy()[0]

        occurrence_probability = prediction["occurrence_probability"].cpu().numpy()[0]

        return format_prediction_output(

            mu,

            sigma,

            sample["past_target"].cpu().numpy(),

            occurrence_probability,

        )

    # ------------------------------------------------------

    def predict_dataset_index(

        self,

        index

    ):
        sample = self.dataset[index]

        return self.predict(sample)


# ==========================================================
# Example
# ==========================================================

if __name__ == "__main__":

    engine = ForecastEngine()

    result = engine.predict_dataset_index(0)

    print_prediction_output(result)
