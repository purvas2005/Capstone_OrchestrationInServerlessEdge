import csv
import numpy as np
import torch
from torch.utils.data import Subset

from .config import PROJECT_ROOT, PILOT_EVALUATION_SAMPLES, RANDOM_SEED
from .inference import ForecastEngine


def main(output_path=None, max_samples=None):
    engine = ForecastEngine()
    dataset = engine.dataset
    _, _, test = dataset.temporal_split()

    # Respect pilot evaluation limit unless overridden
    if max_samples is None and PILOT_EVALUATION_SAMPLES is not None:
        max_samples = PILOT_EVALUATION_SAMPLES

    if max_samples is not None and len(test) > max_samples:
        generator = torch.Generator().manual_seed(RANDOM_SEED + 2)
        selected = torch.randperm(len(test), generator=generator)[:max_samples]
        test = Subset(test, selected.tolist())

    if output_path is None:
        output_path = PROJECT_ROOT / "results" / "predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_index",
            "mean",
            "std",
            "q10",
            "q50",
            "q90",
            "cold_start_risk",
            "warm_capacity",
            "target",
        ])

        for i in range(len(test)):
            sample = test[i]
            result = engine.predict(sample)

            # join horizon arrays with semicolons to keep a single CSV cell
            mean = ";".join(map(str, np.round(result["mean"], 6).tolist()))
            std = ";".join(map(str, np.round(result["std"], 6).tolist()))
            q10 = ";".join(map(str, np.round(result["quantiles"]["q10"], 6).tolist()))
            q50 = ";".join(map(str, np.round(result["quantiles"]["q50"], 6).tolist()))
            q90 = ";".join(map(str, np.round(result["quantiles"]["q90"], 6).tolist()))

            csr = float(result["post_processing"]["cold_start_risk"])
            wc = float(result["post_processing"]["warm_capacity"])

            # target is in log1p space in dataset; convert back
            target_vals = np.round(np.expm1(sample["target"].numpy()).tolist(), 6)
            target = ";".join(map(str, target_vals))

            writer.writerow([i, mean, std, q10, q50, q90, csr, wc, target])

    print(f"Wrote predictions to {output_path}")


if __name__ == "__main__":
    main()
