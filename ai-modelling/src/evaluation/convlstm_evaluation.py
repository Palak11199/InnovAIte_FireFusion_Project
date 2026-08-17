"""
Evaluation script for the ConvLSTM bushfire classifier.

Loads a trained model checkpoint and its associated scaler/metadata bundle,
reconstructs the test split using the exact same data pipeline as training
and reports classification metrics.

Usage:
    python -m src.evaluation.evaluate_convlstm

This script performs no training and never calls scaler.fit(). It is safe to
run repeatedly against the same checkpoint.
"""

import argparse
import json
import os

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from .metrics import flatten_valid, compute_metrics, format_metrics

from ..training.ts_convlstm_forecaster_train import (
    FEATURES,
    GriddedTimeSeriesDataset,
    LABEL_CACHE,
    LABEL_PATH,
    DATA_PATH,
    TRAIN_VAL_RATIO,
    load_and_format_gridded_data,
    load_and_format_label_grid,
)

from ..models.bushfire.ts_convlstm_forecaster import MultivariateTSForecaster

GRID_CACHE_PATH = "src/data/bushfire/data_grid_cache.npy"

CHECKPOINT_PATH = "src/models/bushfire/checkpoints/convlstm_forecaster.pth"
SCALER_PATH = "src/models/bushfire/checkpoints/convlstm_scaler.pkl"

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained ConvLSTM bushfire classifier.")
    p.add_argument("--checkpoint", default=CHECKPOINT_PATH, help="Path to model .pth checkpoint")
    p.add_argument("--scaler-path", default=SCALER_PATH, help="Path to scaler/metadata .pkl bundle")
    p.add_argument("--data-path", default=DATA_PATH, help="Weather CSV path (used only if grid cache is missing)")
    p.add_argument("--label-path", default=LABEL_PATH, help="Satellite detections CSV path (used only if label cache is missing)")
    p.add_argument("--threshold", type=float, default=None, help="Override the fire_threshold stored in the scaler bundle")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default="src/evaluation/eval_results")
    return p.parse_args()


def load_data_and_labels(data_path, label_path, grid_shape_hint=None):
    """Load (or build) the cached weather grid and label grid, unscaled."""
    if os.path.exists(GRID_CACHE_PATH):
        data_grid = np.load(GRID_CACHE_PATH)
    else:
        data_grid = load_and_format_gridded_data(data_path)
        np.save(GRID_CACHE_PATH, data_grid)

    if os.path.exists(LABEL_CACHE):
        label_grid = np.load(LABEL_CACHE)
    else:
        h, w = data_grid.shape[1:3]
        label_grid = load_and_format_label_grid(label_path, data_path, (h, w))
        np.save(LABEL_CACHE, label_grid)

    assert label_grid.shape[:3] == data_grid.shape[:3], (
        f"Grid mismatch: labels {label_grid.shape} vs weather {data_grid.shape}"
    )
    return data_grid, label_grid


def get_test_split(data_grid, label_grid, train_val_ratio=TRAIN_VAL_RATIO):
    """Reproduce the same time-ordered split used at training time."""
    split_idx = int(len(data_grid) * train_val_ratio)
    return data_grid[split_idx:], label_grid[split_idx:]


def scale_and_fill(grid, scaler, n_features):
    shape = grid.shape
    flat = grid.reshape(-1, n_features)
    scaled = scaler.transform(flat)
    scaled[np.isnan(scaled)] = 0.0
    return scaled.reshape(shape)


def run_inference(model, dataloader, device):
    """
    Run the model over a dataloader and collect flattened probability
    predictions and ground truth, without applying any spatial mask yet.

    Outputs:
        tuple: (probs, actuals), each np.ndarray [n_samples, horizon, H, W, 1]
    """
    model.eval()
    all_probs, all_actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            probs = model.predict(X_batch).cpu().numpy()
            all_probs.append(probs)
            all_actuals.append(y_batch.numpy())
    return np.concatenate(all_probs), np.concatenate(all_actuals)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print("STEP 1: Load model checkpoint and scaler bundle")
    model = MultivariateTSForecaster.load(args.checkpoint, map_location=device).to(device)
    bundle = joblib.load(args.scaler_path)
    scaler = bundle["scaler"]
    weather_features = bundle["weather_features"]
    input_steps = bundle["input_steps"]
    horizon = bundle["horizon"]
    grid_shape = bundle["grid_shape"]
    fire_threshold = args.threshold if args.threshold is not None else bundle.get("fire_threshold", 0.5)

    assert weather_features == FEATURES, (
        f"Feature mismatch between bundle ({weather_features}) and current FEATURES ({FEATURES})"
    )
    print(f"Loaded model, config={model.config}")
    print(f"Operating threshold: {fire_threshold}")

    print("STEP 2: Load data and reconstruct test split")
    data_grid, label_grid = load_data_and_labels(args.data_path, args.label_path)
    assert data_grid.shape[1:3] == tuple(grid_shape), (
        f"Grid shape mismatch: data {data_grid.shape[1:3]} vs bundle {grid_shape}"
    )
    valid_mask = ~np.all(np.isnan(data_grid), axis=(0, -1))

    test_grid, test_labels = get_test_split(data_grid, label_grid)
    n_features = len(FEATURES)
    test_scaled = scale_and_fill(test_grid, scaler, n_features)

    print("STEP 3: Build test dataset/dataloader")
    test_dataset = GriddedTimeSeriesDataset(test_scaled, test_labels, input_steps, horizon)
    print(f"Test sequences: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("STEP 4: Run inference")
    probs, actuals = run_inference(model, test_loader, device)

    print("STEP 5: Apply valid-cell mask and flatten")
    y_true, y_prob = flatten_valid(actuals, probs, valid_mask)

    print("STEP 6: Compute metrics")
    metrics = compute_metrics(y_true, y_prob, fire_threshold)
    print(format_metrics(metrics, title="Test Set Metrics"))

    print("STEP 7: Save results")
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    print("EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
