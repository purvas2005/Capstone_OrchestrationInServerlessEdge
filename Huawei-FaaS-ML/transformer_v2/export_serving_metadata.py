"""Export the preprocessing contract required for Transformer serving."""

import json
from pathlib import Path

import duckdb

from .config import (
    DB_PATH,
    FEATURE_TABLE,
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
    TARGET_COLUMN,
    TIME_FEATURES,
)
from .dataset import CATEGORY_MAP, STABILITY_MAP


SCHEMA_VERSION = 1


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


def export_metadata(output_path: str | Path, db_path=None) -> dict:

    db_path = db_path or DB_PATH

    print("=" * 60)
    print("Exporting serving metadata")
    print("=" * 60)

    connection = duckdb.connect(str(db_path))

    if not _table_exists(connection, FEATURE_TABLE):
        raise ValueError(
            f"Expected engineered table `{FEATURE_TABLE}` was not found."
        )

    print(f"Using engineered table: {FEATURE_TABLE}")

    # --------------------------------------------------
    # Vocabulary mappings
    # --------------------------------------------------
    #
    # These reproduce dataset.py exactly:
    #
    # enumerate(sorted(frame["region"].unique()))
    # enumerate(sorted(frame["clusterName"].unique()))
    #
    # Function identity is:
    # region + "::" + clusterName + "::" + funcName
    # --------------------------------------------------

    regions = [
        row[0]
        for row in connection.execute(
            f"""
            SELECT DISTINCT CAST(region AS VARCHAR)
            FROM {FEATURE_TABLE}
            ORDER BY CAST(region AS VARCHAR)
            """
        ).fetchall()
    ]

    clusters = [
        row[0]
        for row in connection.execute(
            f"""
            SELECT DISTINCT CAST(clusterName AS VARCHAR)
            FROM {FEATURE_TABLE}
            ORDER BY CAST(clusterName AS VARCHAR)
            """
        ).fetchall()
    ]

    function_keys = [
        row[0]
        for row in connection.execute(
            f"""
            SELECT DISTINCT
                CAST(region AS VARCHAR)
                || '::' ||
                CAST(clusterName AS VARCHAR)
                || '::' ||
                CAST(funcName AS VARCHAR)
            FROM {FEATURE_TABLE}
            ORDER BY
                CAST(region AS VARCHAR)
                || '::' ||
                CAST(clusterName AS VARCHAR)
                || '::' ||
                CAST(funcName AS VARCHAR)
            """
        ).fetchall()
    ]

    region_map = {
        value: index
        for index, value in enumerate(regions)
    }

    cluster_map = {
        value: index
        for index, value in enumerate(clusters)
    }

    function_map = {
        value: index
        for index, value in enumerate(function_keys)
    }

    # --------------------------------------------------
    # Feature statistics
    # --------------------------------------------------
    #
    # dataset.py uses:
    #
    # feature_frame.mean()
    # feature_frame.std(ddof=0)
    #
    # DuckDB's STDDEV_POP is equivalent to pandas std(ddof=0).
    # --------------------------------------------------

    aggregate_expressions = []

    for feature in PAST_VALUE_FEATURES:
        aggregate_expressions.append(
            f'AVG(CAST("{feature}" AS DOUBLE)) AS "{feature}__mean"'
        )
        aggregate_expressions.append(
            f'STDDEV_POP(CAST("{feature}" AS DOUBLE)) AS "{feature}__std"'
        )

    stats_query = f"""
        SELECT
            {", ".join(aggregate_expressions)}
        FROM {FEATURE_TABLE}
    """

    row = connection.execute(stats_query).fetchone()
    columns = [
        description[0]
        for description in connection.description
    ]

    stats = dict(zip(columns, row))

    feature_mean = {}
    feature_std = {}

    for feature in PAST_VALUE_FEATURES:

        mean_value = stats[f"{feature}__mean"]
        std_value = stats[f"{feature}__std"]

        if mean_value is None:
            mean_value = 0.0

        if std_value is None or std_value == 0:
            std_value = 1.0

        feature_mean[feature] = float(mean_value)
        feature_std[feature] = float(std_value)

    # --------------------------------------------------
    # Build metadata
    # --------------------------------------------------

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "sequence_length": SEQUENCE_LENGTH,
        "prediction_horizon": PREDICTION_HORIZON,
        "past_value_features": list(PAST_VALUE_FEATURES),
        "time_features": list(TIME_FEATURES),
        "target_column": TARGET_COLUMN,
        "target_transform": "log1p",

        "feature_mean": feature_mean,
        "feature_std": feature_std,

        "region_map": {
            str(key): int(value)
            for key, value in region_map.items()
        },

        "cluster_map": {
            str(key): int(value)
            for key, value in cluster_map.items()
        },

        "function_map": {
            str(key): int(value)
            for key, value in function_map.items()
        },

        "category_map": {
            str(key): int(value)
            for key, value in CATEGORY_MAP.items()
        },

        "stability_map": {
            str(key): int(value)
            for key, value in STABILITY_MAP.items()
        },

        "model": {
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "num_encoder_layers": NUM_ENCODER_LAYERS,
            "num_decoder_layers": NUM_DECODER_LAYERS,
            "dim_feedforward": DIM_FEEDFORWARD,
            "dropout": DROPOUT,
            "function_embedding_dim": FUNCTION_EMBED_DIM,
            "num_functions": len(function_map),
            "num_regions": len(region_map),
            "num_clusters": len(cluster_map),
            "num_categories": len(CATEGORY_MAP),
            "num_stability": len(STABILITY_MAP),
        },
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    connection.close()

    print()
    print("Metadata exported successfully")
    print("Functions :", len(function_map))
    print("Regions   :", len(region_map))
    print("Clusters  :", len(cluster_map))
    print("Categories:", len(CATEGORY_MAP))
    print("Stability :", len(STABILITY_MAP))
    print()
    print(f"Output: {output}")

    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--database", default=None)

    args = parser.parse_args()

    export_metadata(
        args.output,
        args.database,
    )
