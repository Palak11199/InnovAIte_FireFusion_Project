# src/inference/run_local_forecast.py
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from api.model_loader import load_models, get_model
from api.inference.bushfire_forecaster import predict_bushfire_forecast  # adjust import path

DATA_PATH = "src/data/bushfire/forecaster_test_data.csv"
GRID_CACHE_PATH = "src/data/bushfire/data_grid_cache.npy"
COORDS_CACHE_PATH = "src/data/bushfire/grid_coords_cache.npz"
OUTPUT_DIR = "src/data/bushfire/forecasts"
CELL_SIZE_DEG = 0.05

MODEL_ID = "bushfire-forecaster-v1"


def get_or_build_coords():
    if os.path.exists(COORDS_CACHE_PATH):
        cached = np.load(COORDS_CACHE_PATH)
        return cached["lats"], cached["lons"]

    print("No coords cache found — extracting lat/lon from CSV (one-time cost)...")
    df = pd.read_csv(DATA_PATH, usecols=[".geo"])

    def extract_coords(geojson_str):
        geojson = json.loads(geojson_str)
        lons, lats = [], []
        for ring in geojson["coordinates"]:
            for point in ring:
                lons.append(float(point[0]))
                lats.append(float(point[1]))
        return min(lons), min(lats)

    lons, lats = [], []
    for s in df[".geo"]:
        try:
            lon, lat = extract_coords(s)
            lons.append(lon)
            lats.append(lat)
        except Exception:
            continue

    unique_lats = np.array(sorted(set(lats)))
    unique_lons = np.array(sorted(set(lons)))
    np.savez(COORDS_CACHE_PATH, lats=unique_lats, lons=unique_lons)
    print(f"Saved coords cache: {len(unique_lats)} lats x {len(unique_lons)} lons")
    return unique_lats, unique_lons


def cell_polygon(lat, lon, size=CELL_SIZE_DEG):
    half = size / 2
    return {
        "type": "Polygon",
        "coordinates": [[
            [lon - half, lat - half],
            [lon + half, lat - half],
            [lon + half, lat + half],
            [lon - half, lat + half],
            [lon - half, lat - half],
        ]],
    }


def build_request(data_grid, unique_lats, unique_lons, valid_mask, input_steps):
    recent = data_grid[-input_steps:]  # [input_steps, H, W, F]
    height, width = recent.shape[1], recent.shape[2]

    features = []
    for row in range(height):
        for col in range(width):
            if not valid_mask[row, col]:
                continue
            obs = np.nan_to_num(recent[:, row, col, :], nan=0.0).tolist()
            features.append({
                "type": "Feature",
                "geometry": cell_polygon(unique_lats[row], unique_lons[col]),
                "properties": {
                    "id": f"cell_{row}_{col}",
                    "observations": obs,
                    "grid_row": row,
                    "grid_col": col,
                },
            })

    return {"type": "FeatureCollection", "features": features}


def main():
    load_models()
    bundle = get_model(MODEL_ID)

    data_grid = np.load(GRID_CACHE_PATH)  # [T, H, W, F], raw/unscaled
    valid_mask = ~np.all(np.isnan(data_grid), axis=(0, -1))
    unique_lats, unique_lons = get_or_build_coords()

    input_steps = bundle.metadata.get("input_steps", 60)
    request_geojson = build_request(data_grid, unique_lats, unique_lons, valid_mask, input_steps)
    print(f"Built request with {len(request_geojson['features'])} cells")

    result = predict_bushfire_forecast(request_geojson, bundle)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(OUTPUT_DIR, f"forecast_{ts}.geojson")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved forecast to {out_path}")


if __name__ == "__main__":
    main()