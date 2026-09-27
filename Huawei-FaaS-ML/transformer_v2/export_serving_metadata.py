"""Export the preprocessing contract required for Transformer serving."""

import json
from pathlib import Path

from .config import (
    CATEGORY_MAP,
    D_MODEL,
    DIM_FEEDFORWARD,
    DROPOUT,
    FUNCTION_EMBED_DIM,
    NUM_DECODER_LAYERS,
    NUM_ENCODER_LAYERS,
    NHEAD,
    PAST_VALUE_FEATURES,
    PREDICTION_HORIZON,
    SEQUENCE_LENGTH,
    STABILITY_MAP,
    TARGET_COLUMN,
    TIME_FEATURES,
)
from .dataset import HuaweiForecastDataset


SCHEMA_VERSION = 1


def export_metadata(output_path: str | Path, db_path=None) -> dict:
    """Build and persist the exact preprocessing metadata used by training."""
    dataset = HuaweiForecastDataset(db_path=db_path) if db_path is not None else HuaweiForecastDataset()

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "sequence_length": SEQUENCE_LENGTH,
        "prediction_horizon": PREDICTION_HORIZON,
        "past_value_features": list(PAST_VALUE_FEATURES),
        "time_features": list(TIME_FEATURES),
        "target_column": TARGET_COLUMN,
        "target_transform": dataset.target_transform,
        "feature_mean": {key: float(value) for key, value in dataset.feature_mean.items()},
        "feature_std": {key: float(value) for key, value in dataset.feature_std.items()},
        "region_map": {str(key): int(value) for key, value in dataset.region_map.items()},
        "cluster_map": {str(key): int(value) for key, value in dataset.cluster_map.items()},
        "function_map": {str(key): int(value) for key, value in dataset.function_map.items()},
        "category_map": {str(key): int(value) for key, value in CATEGORY_MAP.items()},
        "stability_map": {str(key): int(value) for key, value in STABILITY_MAP.items()},
        "model": {
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "num_encoder_layers": NUM_ENCODER_LAYERS,
            "num_decoder_layers": NUM_DECODER_LAYERS,
            "dim_feedforward": DIM_FEEDFORWARD,
            "dropout": DROPOUT,
            "function_embedding_dim": FUNCTION_EMBED_DIM,
            "num_functions": dataset.num_functions,
            "num_regions": dataset.num_regions,
            "num_clusters": dataset.num_clusters,
            "num_categories": dataset.num_categories,
            "num_stability": dataset.num_stability,
        },
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--database", default=None)
    args = parser.parse_args()
    export_metadata(args.output, args.database)