"""Phase 3: classifier training, calibration, and abstention-threshold
evaluation for the RAG abstention classifier.

Consumes Phase 2's `train_features.parquet` / `eval_features.parquet`
(see `abstention_features.pipeline` for how those columns are produced)
-- this package is pure `numpy`/`pandas`/`scikit-learn` over already
-extracted features, no embedding models, no GPU, no network access
needed. See `features.py`, `train.py`, `calibrate.py`, `evaluate.py`, and
`scripts/train_and_evaluate.py` for the individual pieces.
"""
