from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import DB_PATH

# ==========================================================
# Configuration
# ==========================================================

SEQUENCE_LENGTH = 60
PREDICTION_HORIZON = 10

FEATURE_TABLE = "feature_table"

# ----------------------------------------------------------
# Numerical features used by Transformer
# ----------------------------------------------------------

NUMERIC_FEATURES = [

    "requests",

    "avg_cpu",
    "avg_memory",
    "avg_runtime",
    "avg_request_size",

    "active_pods",
    "active_users",

    "lag_1",
    "lag_5",
    "lag_15",
    "lag_30",

    "rolling_mean_5",
    "rolling_mean_15",
    "rolling_mean_30",

    "rolling_std",

    "rolling_max",
    "rolling_min",
    "rolling_median",

    "warm_capacity",

    "trend",
    "growth_rate",

    "coeff_variation",

    "cpu_per_request",
    "memory_per_request",
    "runtime_per_request",

    "warm_ratio",

    "hour_sin",
    "hour_cos",

    "minute_sin",
    "minute_cos"
]

# ----------------------------------------------------------
# Metadata encodings
# ----------------------------------------------------------

CATEGORY_MAP = {
    "Rare": 0,
    "Normal": 1,
    "Hot": 2
}

STABILITY_MAP = {
    "Stable": 0,
    "Noisy": 1
}


class HuaweiSequenceDataset(Dataset):

    def __init__(self):

        print("Loading feature table from DuckDB...")

        self.con = duckdb.connect(str(DB_PATH))

        self.samples = []

        # --------------------------------------------------
        # Load table
        # --------------------------------------------------

        df = self.con.execute(f"""

        SELECT *

        FROM {FEATURE_TABLE}

        ORDER BY

            region,
            clusterName,
            funcName,
            minute

        """).df()

        print(f"Loaded {len(df):,} rows")

        # --------------------------------------------------
        # Encode metadata
        # --------------------------------------------------

        df["category_id"] = df["category"].map(CATEGORY_MAP)

        df["stability_id"] = df["stability"].map(STABILITY_MAP)

        # --------------------------------------------------
        # Replace NaNs produced by rolling statistics
        # --------------------------------------------------

        df = df.fillna(0)

        # --------------------------------------------------
        # Build sequences
        # --------------------------------------------------

        grouped = df.groupby(
            [
                "region",
                "clusterName",
                "funcName"
            ],
            sort=False
        )

        total_sequences = 0

        for key, group in grouped:

            group = group.reset_index(drop=True)

            if len(group) < SEQUENCE_LENGTH + PREDICTION_HORIZON:
                continue

            features = group[
                NUMERIC_FEATURES
            ].values.astype(np.float32)

            category = group["category_id"].iloc[0]

            stability = group["stability_id"].iloc[0]

            target = group["requests"].values.astype(np.float32)

            for start in range(
                0,
                len(group) - SEQUENCE_LENGTH - PREDICTION_HORIZON + 1
            ):

                end = start + SEQUENCE_LENGTH

                future = end + PREDICTION_HORIZON

                x = features[start:end]

                y = np.log1p(target[end:future])

                self.samples.append(
                    (
                        x,
                        y,
                        category,
                        stability
                    )
                )

                total_sequences += 1

        print(f"Created {total_sequences:,} training sequences")

    # ----------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    # ----------------------------------------------------------

    def __getitem__(self, idx):

        x, y, category, stability = self.samples[idx]

        return {

            "x": torch.tensor(x, dtype=torch.float32),

            "y": torch.from_numpy(y.astype(np.float32)),

            "category": torch.tensor(category),

            "stability": torch.tensor(stability)

        }


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    dataset = HuaweiSequenceDataset()

    print()

    print("=" * 60)
    print("Dataset Summary")
    print("=" * 60)

    print("Sequences :", len(dataset))

    sample = dataset[0]

    print()

    print("Input Shape")
    print(sample["x"].shape)

    print()

    print("Target Shape")
    print(sample["y"].shape)

    print()

    print("Category")
    print(sample["category"])

    print()

    print("Stability")
    print(sample["stability"])

