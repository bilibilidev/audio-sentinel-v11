# Causal CNN v1 candidate weights

These six NPZ files are the completed CPU experiment of 2026-10-17. They contain arrays only; load with `allow_pickle=False`. SHA256 values are in `sha256.json`.

- `fold_1.npz` through `fold_5.npz`: each trains on the other four official ESC-50 folds, with train-only normalization.
- `final_all_esc50.npz`: trained on all 390 selected ESC-50 recordings, not used as a held-out CV model.

No external evaluation recordings were used for training. Exported final weights are in `simulator/cnn/cnn_weights.h`. The original v11 MLP NPZ and application headers are retained. See [usage, measurements and limitations](../../docs/cnn-v1.md) and the repository's third-party notices for training-data and model-use boundaries.
