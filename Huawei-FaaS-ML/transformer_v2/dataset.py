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
    "Hot": 2,
}

STABILITY_MAP = {
    "Stable": 0,
    "Noisy": 1,
}

RAW_TIME_COLUMNS = ("time_worker", "time_frontend")


# ==========================================================
# Helpers
# ==========================================================

def _table_exists(connection, table_name):

    return bool(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = ?
            """,
            [table_name],
        ).fetchone()[0]
    )


def _coerce_time_series(series):

    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().sum() == 0:
        return pd.to_datetime(series, errors="coerce", utc=True)

    max_abs = float(numeric.dropna().abs().max())

    if max_abs >= 1e17:
        unit = "ns"
    elif max_abs >= 1e14:
        unit = "us"
    elif max_abs >= 1e11:
        unit = "ms"
    else:
        unit = "s"

    return pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)


def _add_missing_columns(frame, columns, default=0.0):

    for column in columns:
        if column not in frame.columns:
            frame[column] = default

    return frame


def _select_numeric_column(frame, candidates, default=0.0):

    for candidate in candidates:
        if candidate in frame.columns:
            return pd.to_numeric(frame[candidate], errors="coerce").fillna(default)

    return pd.Series(default, index=frame.index, dtype=np.float32)


# ==========================================================
# Dataset
# ==========================================================

class HuaweiForecastDataset(Dataset):
    """
    Dataset used by the Transformer.

    It can read either:

    - the engineered `feature_table`, or
    - the raw Huawei trace table `requests`

    The raw trace path aggregates rows into minute buckets and derives the
    same engineered features expected by the model.
    """

    def __init__(self, db_path=DB_PATH):

        super().__init__()

        print("=" * 60)
        print("Loading dataset")
        print("=" * 60)

        self.connection = duckdb.connect(str(db_path))

        if _table_exists(self.connection, FEATURE_TABLE):
            print(f"Using engineered table: {FEATURE_TABLE}")
            self._loaded_from_raw = False
            frame = self.connection.execute(
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
        elif _table_exists(self.connection, "requests"):
            print("Using raw Huawei trace table: requests")
            self._loaded_from_raw = True
            frame = self.connection.execute(
                "SELECT * FROM requests"
            ).df()
            frame = self._build_feature_frame_from_raw(frame)
        else:
            raise ValueError(
                "No usable table found. Expected either `feature_table` or `requests` in the DuckDB database."
            )

        if "region" not in frame.columns:
            frame["region"] = "global"

        frame = frame.fillna(0)

        # --------------------------------------------------
        # Encode / normalize identifiers
        # --------------------------------------------------

        frame["region"] = frame["region"].astype(str)
        frame["clusterName"] = frame["clusterName"].astype(str)
        frame["funcName"] = frame["funcName"].astype(str)

        self.region_map = {
            value: index for index, value in enumerate(sorted(frame["region"].unique()))
        }

        self.cluster_map = {
            value: index for index, value in enumerate(sorted(frame["clusterName"].unique()))
        }

        self.function_map = {
            value: index for index, value in enumerate(sorted(frame["funcName"].unique()))
        }

        frame["region"] = frame["region"].map(self.region_map).astype(np.int64)
        frame["clusterName"] = frame["clusterName"].map(self.cluster_map).astype(np.int64)
        frame["funcName"] = frame["funcName"].map(self.function_map).astype(np.int64)

        # --------------------------------------------------
        # Metadata labels
        # --------------------------------------------------

        if self._loaded_from_raw or "category" not in frame.columns or frame["category"].dtype == object:
            frame["category"] = self._derive_categories(frame)
        else:
            frame["category"] = pd.to_numeric(frame["category"], errors="coerce").fillna(0).astype(np.int64)

        if self._loaded_from_raw or "stability" not in frame.columns or frame["stability"].dtype == object:
            frame["stability"] = self._derive_stability(frame)
        else:
            frame["stability"] = pd.to_numeric(frame["stability"], errors="coerce").fillna(0).astype(np.int64)

        # --------------------------------------------------
        # Ensure model features exist and are numeric
        # --------------------------------------------------

        frame = _add_missing_columns(frame, PAST_VALUE_FEATURES, default=0.0)
        frame = _add_missing_columns(frame, TIME_FEATURES, default=0.0)

        for column in PAST_VALUE_FEATURES + TIME_FEATURES + [TARGET_COLUMN, "minute"]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame = frame.fillna(0)
        frame["raw_target"] = frame[TARGET_COLUMN].astype(np.float32)
        frame = frame.astype(
            {column: np.float32 for column in PAST_VALUE_FEATURES}
        )

        # --------------------------------------------------
        # Normalize numerical inputs
        # --------------------------------------------------

        feature_frame = frame[PAST_VALUE_FEATURES].astype(np.float32)

        self.feature_mean = feature_frame.mean()
        self.feature_std = feature_frame.std(ddof=0).replace(0, 1.0)

        frame.loc[:, PAST_VALUE_FEATURES] = (
            (feature_frame - self.feature_mean) / self.feature_std
        ).to_numpy(dtype=np.float32)

        self.target_transform = "log1p"

        # --------------------------------------------------
        # Vocabulary sizes
        # --------------------------------------------------

        self.num_functions = len(self.function_map)
        self.num_regions = len(self.region_map)
        self.num_clusters = len(self.cluster_map)
        self.num_categories = len(CATEGORY_MAP)
        self.num_stability = len(STABILITY_MAP)

        print()
        print("Vocabulary")
        print("Functions :", self.num_functions)
        print("Regions   :", self.num_regions)
        print("Clusters  :", self.num_clusters)
        print("Categories:", self.num_categories)
        print("Stability :", self.num_stability)

        # --------------------------------------------------
        # Build sequences
        # --------------------------------------------------

        self.groups = []
        self.samples = []

        grouped = frame.groupby(
            ["region", "clusterName", "funcName"],
            sort=False,
        )

        print(f"Groups : {len(grouped)}")

        for _, group in grouped:

            group = group.sort_values("minute").reset_index(drop=True)

            if len(group) < SEQUENCE_LENGTH + PREDICTION_HORIZON:
                continue

            values = group[PAST_VALUE_FEATURES].to_numpy(dtype=np.float32)
            past_time = group[TIME_FEATURES].to_numpy(dtype=np.float32)
            target = np.log1p(group["raw_target"].to_numpy(dtype=np.float32))

            static = {
                "function": int(group["funcName"].iloc[0]),
                "region": int(group["region"].iloc[0]),
                "cluster": int(group["clusterName"].iloc[0]),
                "category": int(group["category"].iloc[0]),
                "stability": int(group["stability"].iloc[0]),
            }

            group_data = {
                "values": values,
                "time": past_time,
                "target": target,
                "static": static,
            }

            group_id = len(self.groups)
            self.groups.append(group_data)

            max_start = len(group) - SEQUENCE_LENGTH - PREDICTION_HORIZON + 1

            for start in range(max_start):
                self.samples.append((group_id, start))

        print(f"Sequences : {len(self.samples):,}")

    # ======================================================
    # Raw Huawei trace -> feature table
    # ======================================================

    def _build_feature_frame_from_raw(self, frame):

        frame = frame.copy()

        if "clusterName" not in frame.columns or "funcName" not in frame.columns:
            raise ValueError(
                "Raw Huawei trace must include `clusterName` and `funcName` columns."
            )

        frame["clusterName"] = frame["clusterName"].astype(str)
        frame["funcName"] = frame["funcName"].astype(str)

        if "region" not in frame.columns:
            frame["region"] = "global"
        else:
            frame["region"] = frame["region"].astype(str)

        time_column = None
        for candidate in RAW_TIME_COLUMNS:
            if candidate in frame.columns:
                time_column = candidate
                break

        if time_column is None:
            raise ValueError(
                "Raw Huawei trace must include either `time_worker` or `time_frontend`."
            )

        timestamps = _coerce_time_series(frame[time_column])

        if timestamps.isna().all():
            raise ValueError(
                f"Could not parse timestamps from `{time_column}`."
            )

        timestamps = timestamps.fillna(method="ffill").fillna(method="bfill")
        minute = ((timestamps - timestamps.min()).dt.total_seconds() // 60).astype(np.int64)
        frame["minute"] = minute

        requests = frame.copy()
        requests["requests"] = 1

        aggregated = requests.groupby(
            ["minute", "region", "clusterName", "funcName"],
            as_index=False,
        ).agg(
            requests=("requests", "sum"),
            avg_cpu=("cpu_usage", "mean") if "cpu_usage" in requests.columns else ("requests", "mean"),
            avg_memory=("memory_usage", "mean") if "memory_usage" in requests.columns else ("requests", "mean"),
            avg_runtime=("runtimeCost", "mean") if "runtimeCost" in requests.columns else ("requests", "mean"),
            avg_request_size=("requestBodySize", "mean") if "requestBodySize" in requests.columns else ("requests", "mean"),
            active_pods=("podID", pd.Series.nunique) if "podID" in requests.columns else ("requests", "sum"),
            active_users=("userID", pd.Series.nunique) if "userID" in requests.columns else ("requests", "sum"),
        )

        # Optional raw cost signals if they are present in the Huawei trace.
        optional_means = {
            "workerCost": "avg_worker_cost",
            "frontendCost": "avg_frontend_cost",
            "busCost": "avg_bus_cost",
            "readBodyCost": "avg_read_body_cost",
            "writeRspCost": "avg_write_rsp_cost",
            "totalCost_worker": "avg_total_cost_worker",
            "totalCost_frontend": "avg_total_cost_frontend",
        }

        for source, target in optional_means.items():
            if source in requests.columns:
                aggregated[target] = requests.groupby(
                    ["minute", "region", "clusterName", "funcName"]
                )[source].mean().values
            else:
                aggregated[target] = 0.0

        aggregated = aggregated.sort_values(
            ["region", "clusterName", "funcName", "minute"]
        ).reset_index(drop=True)

        aggregated["hour_of_day"] = ((aggregated["minute"] // 60) % 24).astype(np.float32)
        aggregated["minute_of_hour"] = (aggregated["minute"] % 60).astype(np.float32)
        aggregated["hour_sin"] = np.sin(2.0 * np.pi * aggregated["hour_of_day"] / 24.0)
        aggregated["hour_cos"] = np.cos(2.0 * np.pi * aggregated["hour_of_day"] / 24.0)
        aggregated["minute_sin"] = np.sin(2.0 * np.pi * aggregated["minute_of_hour"] / 60.0)
        aggregated["minute_cos"] = np.cos(2.0 * np.pi * aggregated["minute_of_hour"] / 60.0)

        feature_rows = []

        grouped = aggregated.groupby(
            ["region", "clusterName", "funcName"],
            sort=False,
        )

        for _, group in grouped:

            group = group.sort_values("minute").reset_index(drop=True)

            if len(group) < SEQUENCE_LENGTH + PREDICTION_HORIZON:
                continue

            requests_series = group["requests"].astype(np.float32)

            engineered = group.copy()

            for lag in (1, 5, 15, 30):
                engineered[f"lag_{lag}"] = requests_series.shift(lag)

            for window in (5, 15, 30):
                rolling = requests_series.rolling(window=window, min_periods=1)
                engineered[f"rolling_mean_{window}"] = rolling.mean()

            rolling_30 = requests_series.rolling(window=30, min_periods=1)
            engineered["rolling_std"] = rolling_30.std(ddof=0).fillna(0)
            engineered["rolling_max"] = rolling_30.max()
            engineered["rolling_min"] = rolling_30.min()
            engineered["rolling_median"] = rolling_30.median()
            engineered["warm_capacity"] = rolling_30.quantile(0.90)

            engineered["trend"] = requests_series.diff().fillna(0)
            engineered["growth_rate"] = requests_series.pct_change().replace([np.inf, -np.inf], 0).fillna(0)

            rolling_mean_30 = engineered["rolling_mean_30"].replace(0, np.nan)
            engineered["coeff_variation"] = (
                engineered["rolling_std"] / rolling_mean_30
            ).replace([np.inf, -np.inf], 0).fillna(0)

            for numerator, target in (
                ("avg_cpu", "cpu_per_request"),
                ("avg_memory", "memory_per_request"),
                ("avg_runtime", "runtime_per_request"),
            ):
                engineered[target] = np.where(
                    requests_series.to_numpy() == 0,
                    0.0,
                    engineered[numerator].to_numpy() / requests_series.to_numpy(),
                )

            engineered["warm_ratio"] = np.where(
                engineered["warm_capacity"].to_numpy() == 0,
                0.0,
                requests_series.to_numpy() / engineered["warm_capacity"].to_numpy(),
            )

            feature_rows.append(engineered)

        if not feature_rows:
            raise ValueError(
                "Raw Huawei trace did not produce any usable sequences. Check the time granularity and grouping keys."
            )

        frame = pd.concat(feature_rows, ignore_index=True)

        required_columns = ["region", "clusterName", "funcName", "category", "stability"]
        for column in required_columns:
            if column not in frame.columns:
                frame[column] = 0

        return frame

    # ======================================================
    # Metadata
    # ======================================================

    def _derive_categories(self, frame):

        function_counts = frame.groupby("funcName")["requests"].sum()
        upper = function_counts.quantile(0.80)
        lower = function_counts.quantile(0.25)

        function_total_map = frame.groupby("funcName")["requests"].sum().to_dict()

        categories = []
        for function in frame["funcName"]:
            total_requests = function_total_map.get(function, 0)
            if total_requests >= upper:
                categories.append(CATEGORY_MAP["Hot"])
            elif total_requests <= lower:
                categories.append(CATEGORY_MAP["Rare"])
            else:
                categories.append(CATEGORY_MAP["Normal"])

        return pd.Series(categories, index=frame.index, dtype=np.int64)

    def _derive_stability(self, frame):

        grouped = frame.groupby("funcName")
        stats = grouped["requests"].agg(["mean", "std"])
        stats["std"] = stats["std"].fillna(0)
        stats["mean"] = stats["mean"].replace(0, np.nan)
        coeff = (stats["std"] / stats["mean"]).replace([np.inf, -np.inf], np.nan).fillna(0)
        stable_functions = coeff <= 1.0
        function_stability = {
            func_name: STABILITY_MAP["Stable"] if stable_functions.loc[func_name] else STABILITY_MAP["Noisy"]
            for func_name in stats.index
        }

        return frame["funcName"].map(function_stability).fillna(STABILITY_MAP["Noisy"]).astype(np.int64)

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
            "past_values": torch.tensor(past_values, dtype=torch.float32),
            "past_time_features": torch.tensor(past_time, dtype=torch.float32),
            "future_time_features": torch.tensor(future_time, dtype=torch.float32),
            "function": torch.tensor(static["function"], dtype=torch.long),
            "region": torch.tensor(static["region"], dtype=torch.long),
            "cluster": torch.tensor(static["cluster"], dtype=torch.long),
            "category": torch.tensor(static["category"], dtype=torch.long),
            "stability": torch.tensor(static["stability"], dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.float32),
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
            "stability": self.num_stability,
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
    print(dataset.get_vocab_sizes())
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

    assert sample["past_values"].shape == (SEQUENCE_LENGTH, len(PAST_VALUE_FEATURES))
    assert sample["past_time_features"].shape == (SEQUENCE_LENGTH, len(TIME_FEATURES))
    assert sample["future_time_features"].shape == (PREDICTION_HORIZON, len(TIME_FEATURES))
    assert sample["target"].shape == (PREDICTION_HORIZON,)

    print("✓ All dataset checks passed.")
    print()
    print("Dataset is ready for training.")
