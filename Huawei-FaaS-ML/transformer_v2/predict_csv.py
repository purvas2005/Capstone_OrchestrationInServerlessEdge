import csv
import numpy as np
import torch
from torch.utils.data import Subset

from .config import PROJECT_ROOT, PILOT_EVALUATION_SAMPLES, RANDOM_SEED
from .inference import ForecastEngine
from .config import PREDICTION_HORIZON

# Number of future minutes to include in the CSV (first N timesteps)
N_MINUTES = 2


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

    # invert function map to get human-readable function key
    inverse_function_map = {v: k for k, v in dataset.function_map.items()}

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_index",
            "function_idx",
            "function_key",
            "minute_offset",
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

            func_idx = int(sample["function"].item())
            func_key = inverse_function_map.get(func_idx, "unknown")

            # ensure arrays
            mean_arr = np.asarray(result["mean"])
            std_arr = np.asarray(result["std"])
            q10_arr = np.asarray(result["quantiles"]["q10"])
            q50_arr = np.asarray(result["quantiles"]["q50"])
            q90_arr = np.asarray(result["quantiles"]["q90"])
            csr_arr = np.asarray(result["post_processing"]["cold_start_risk"])
            wc = float(result["post_processing"]["warm_capacity"])

            # number of timesteps to emit (cap by available horizon)
            n_emit = min(N_MINUTES, len(mean_arr), PREDICTION_HORIZON)

            target_vals = np.round(np.expm1(sample["target"].numpy()).tolist(), 6)

            for offset in range(n_emit):
                mean = float(np.round(mean_arr[offset], 6))
                std = float(np.round(std_arr[offset], 6))
                q10 = float(np.round(q10_arr[offset], 6))
                q50 = float(np.round(q50_arr[offset], 6))
                q90 = float(np.round(q90_arr[offset], 6))
                csr = float(np.round(csr_arr[offset], 6)) if csr_arr.size > 1 else float(np.round(csr_arr.item(), 6))
                target = target_vals[offset] if offset < len(target_vals) else ""

                writer.writerow([
                    i,
                    func_idx,
                    func_key,
                    offset + 1,
                    mean,
                    std,
                    q10,
                    q50,
                    q90,
                    csr,
                    wc,
                    target,
                ])

    print(f"Wrote predictions to {output_path}")


if __name__ == "__main__":
    main()
