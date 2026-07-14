from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from .config import *


# ==========================================================
# Metadata Encodings
# ==========================================================

CATEGORY_MAP = {
    "Rare": 0,
    "Normal": 1,
    "Hot": 2
}

STABILITY_MAP = {
    "Stable": 0,
    "Noisy": 1
}


class HuaweiForecastDataset(Dataset):
    """
    Dataset used by the Transformer.

    Every sample contains

        Past numerical features

        Past time features

        Future time features

        Static metadata

        Future target sequence

    Shapes

        past_values
            (60, num_features)

        past_time_features
            (60, 4)

        future_time_features
            (10, 4)

        target
            (10,)
    """

    def __init__(self):

        super().__init__()

        print("=" * 60)
        print("Loading feature table")
        print("=" * 60)

        self.connection = duckdb.connect(str(DB_PATH))

        self.df = self.connection.execute(
            f"""
            SELECT *
            FROM {FEATURE_TABLE}
            ORDER BY
                region,
                clusterName,
                funcName,
                minute
            """
        ).df()

        print(f"Rows : {len(self.df):,}")

        # --------------------------------------------------
        # Replace NaNs
        # --------------------------------------------------

        self.df = self.df.fillna(0)

        # --------------------------------------------------
        # Encode metadata
        # --------------------------------------------------

        self.df["category"] = (
            self.df["category"]
            .map(CATEGORY_MAP)
            .astype(np.int64)
        )

        self.df["stability"] = (
            self.df["stability"]
            .map(STABILITY_MAP)
            .astype(np.int64)
        )

        # --------------------------------------------------
        # Encode region
        # --------------------------------------------------

        unique_regions = sorted(
            self.df["region"].unique()
        )

        self.region_map = {
            value: index
            for index, value in enumerate(unique_regions)
        }

        self.df["region"] = (
            self.df["region"]
            .map(self.region_map)
            .astype(np.int64)
        )

        # --------------------------------------------------
        # Vocabulary sizes
        # --------------------------------------------------

        self.num_functions = (
            int(self.df["funcName"].max()) + 1
        )

        self.num_regions = (
            len(self.region_map)
        )

        self.num_clusters = (
            int(self.df["clusterName"].max()) + 1
        )

        self.num_categories = (
            len(CATEGORY_MAP)
        )

        self.num_stability = (
            len(STABILITY_MAP)
        )

        print()
        print("Vocabulary")

        print("Functions :", self.num_functions)
        print("Regions   :", self.num_regions)
        print("Clusters  :", self.num_clusters)

        # --------------------------------------------------
        # Storage
        # --------------------------------------------------

        self.groups = []

        self.samples = []

        grouped = self.df.groupby(
            [
                "region",
                "clusterName",
                "funcName"
            ],
            sort=False
        )

        print(f"Groups : {len(grouped)}")

        # --------------------------------------------------
        # Build Groups
        # --------------------------------------------------

        for _, group in grouped:

            group = (
                group
                .sort_values("minute")
                .reset_index(drop=True)
            )

            if len(group) < (
                SEQUENCE_LENGTH +
                PREDICTION_HORIZON
            ):
                continue

            values = group[
                PAST_VALUE_FEATURES
            ].values.astype(np.float32)

            past_time = group[
                TIME_FEATURES
            ].values.astype(np.float32)

            target = np.log1p(
                group[TARGET_COLUMN]
                .values
                .astype(np.float32)
            )

            static = {

                "function":
                    int(group["funcName"].iloc[0]),

                "region":
                    int(group["region"].iloc[0]),

                "cluster":
                    int(group["clusterName"].iloc[0]),

                "category":
                    int(group["category"].iloc[0]),

                "stability":
                    int(group["stability"].iloc[0])

            }

            group_data = {

                "values": values,

                "time": past_time,

                "target": target,

                "static": static

            }

            group_id = len(self.groups)

            self.groups.append(group_data)
            # ------------------------------------------
            # Sliding Window
            # ------------------------------------------

            max_start = (
                len(group)
                - SEQUENCE_LENGTH
                - PREDICTION_HORIZON
                + 1
            )

            for start in range(max_start):

                self.samples.append(
                    (
                        group_id,
                        start
                    )
                )

        print(f"Sequences : {len(self.samples):,}")

    # ======================================================
    # Dataset Length
    # ======================================================

    def __len__(self):

        return len(self.samples)

    # ======================================================
    # Get Sample
    # ======================================================

    def __getitem__(self, index):

        group_id, start = self.samples[index]

        group = self.groups[group_id]

        end = start + SEQUENCE_LENGTH

        future = end + PREDICTION_HORIZON

        past_values = group["values"][start:end]

        past_time = group["time"][start:end]

        future_time = group["time"][end:future]

        target = group["target"][end:future]

        static = group["static"]

        return {

            "past_values": torch.tensor(
                past_values,
                dtype=torch.float32
            ),

            "past_time_features": torch.tensor(
                past_time,
                dtype=torch.float32
            ),

            "future_time_features": torch.tensor(
                future_time,
                dtype=torch.float32
            ),

            "function": torch.tensor(
                static["function"],
                dtype=torch.long
            ),

            "region": torch.tensor(
                static["region"],
                dtype=torch.long
            ),

            "cluster": torch.tensor(
                static["cluster"],
                dtype=torch.long
            ),

            "category": torch.tensor(
                static["category"],
                dtype=torch.long
            ),

            "stability": torch.tensor(
                static["stability"],
                dtype=torch.long
            ),

            "target": torch.tensor(
                target,
                dtype=torch.float32
            )

        }
# ======================================================
# Convenience Methods
# ======================================================

def get_num_features(self):

    return len(PAST_VALUE_FEATURES)


def get_prediction_horizon(self):

    return PREDICTION_HORIZON


def get_sequence_length(self):

    return SEQUENCE_LENGTH


def get_vocab_sizes(self):

    return {

        "functions": self.num_functions,

        "regions": self.num_regions,

        "clusters": self.num_clusters,

        "categories": self.num_categories,

        "stability": self.num_stability

    }


# ======================================================
# Dataset Verification
# ======================================================

if __name__ == "__main__":

    dataset = HuaweiForecastDataset()

    print()

    print("=" * 60)
    print("Dataset Summary")
    print("=" * 60)

    print(f"Sequences : {len(dataset):,}")

    print()

    print("Vocabulary")

    vocab = dataset.get_vocab_sizes()

    print(vocab)

    print()

    sample = dataset[0]

    print("Keys")

    print(sample.keys())

    print()

    print("Past Values")

    print(sample["past_values"].shape)

    print()

    print("Past Time Features")

    print(sample["past_time_features"].shape)

    print()

    print("Future Time Features")

    print(sample["future_time_features"].shape)

    print()

    print("Target")

    print(sample["target"].shape)

    print()

    print("Function")

    print(sample["function"])

    print()

    print("Region")

    print(sample["region"])

    print()

    print("Cluster")

    print(sample["cluster"])

    print()

    print("Category")

    print(sample["category"])

    print()

    print("Stability")

    print(sample["stability"])

    print()

    print("First Past Feature Vector")

    print(sample["past_values"][0])

    print()

    print("Target")

    print(sample["target"])
    print()

    print("=" * 60)
    print("Sanity Checks")
    print("=" * 60)

    assert sample["past_values"].shape == (
        SEQUENCE_LENGTH,
        len(PAST_VALUE_FEATURES)
    )

    assert sample["past_time_features"].shape == (
        SEQUENCE_LENGTH,
        len(TIME_FEATURES)
    )

    assert sample["future_time_features"].shape == (
        PREDICTION_HORIZON,
        len(TIME_FEATURES)
    )

    assert sample["target"].shape == (
        PREDICTION_HORIZON,
    )

    print("✓ All dataset checks passed.")

    print()

    print("Dataset is ready for training.")
