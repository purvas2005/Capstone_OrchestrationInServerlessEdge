# Dense minute feature pipeline

This pipeline converts the event-level Huawei trace into a per-function,
per-minute forecasting table.  It preserves the sparse aggregation as
`aggregated_requests` and creates `aggregated_requests_dense`, which has one
row for every minute of day 30 for each observed `(region, clusterName,
funcName)` group.  Missing event rows are represented as `requests = 0`.

Run these commands from the `Huawei-FaaS-ML` directory, in this order:

```bash
python -m feature_engineering.aggregate
python -m feature_engineering.build_function_metadata
python -m feature_engineering.build_feature_table
```

`aggregate` stops if any group does not contain the complete minute calendar.
Only then run the bounded Transformer pilot:

```bash
python -m transformer_v2.train
python -m transformer_v2.evaluate
```

The dense calendar assumes that, for an included function group, absence of a
request record means zero requests.  Do not interpret unrecorded portions of
an externally subsampled region as real zeros without confirming the trace
sampling semantics.
