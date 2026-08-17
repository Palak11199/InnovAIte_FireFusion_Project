"""
Bushfire ConvLSTM Forecaster Inference: pure functions over a LoadedModel bundle.
Routers call this after resolving model_id via model_loader.
"""
from typing import Any, Optional, Tuple
import numpy as np
import torch

from api.schemas.bushfire import ForecastRequest, ForecastResponse, GeoFeatureOut, ForecastPropertiesOut, DEFAULT_FEATURE_NAMES, RISK_LEVEL_LABELS, prob_to_risk_level
from api.model_loader import LoadedModel


def predict_bushfire_forecast(geojson_dict: dict, bundle: LoadedModel) -> dict:
    """
    Predict fire-occurrence probability from weather and environmental time-series observations using the ConvLSTM fire forecaster model.
    
    Args:
        geojson_dict: GeoJSON FeatureCollection with weather
        bundle: LoadedModel bundle with model, device, scaler, metadata
    
    Returns:
        GeoJSON FeatureCollection with fire probability and threshold prediction per cell
    """
    # Validate input
    request = ForecastRequest(**geojson_dict)
    
    # Extract observations and metadata from input features
    observations_list = []
    feature_metas = []
    
    input_steps = bundle.metadata.get("input_steps", 60)
    horizon = bundle.metadata.get("horizon", 1)
    grid_shape = bundle.metadata.get("grid_shape")
    
    # Collect observations from each feature
    for feature in request.features:
        # [seq_len, n_features]
        obs = np.array(feature.properties.observations, dtype=np.float32)
        
        # Enforce sequence length: pad or truncate
        seq_len = obs.shape[0]
        if seq_len < input_steps:
            # Pad with zeros at the beginning (oldest timesteps)
            pad_width = ((input_steps - seq_len, 0), (0, 0))
            obs = np.pad(obs, pad_width, mode="constant", constant_values=0.0)
        elif seq_len > input_steps:
            # Truncate to most recent timesteps
            obs = obs[-input_steps:, :]
        
        observations_list.append(obs)
        feature_metas.append({
            "id": feature.properties.id,
            "geometry": feature.geometry,
            "grid_row": feature.properties.__dict__.get("grid_row"),
            "grid_col": feature.properties.__dict__.get("grid_col"),
        })

    expected_features = bundle.metadata.get("weather_features", DEFAULT_FEATURE_NAMES)
    if observations_list:
        n_features = observations_list[0].shape[1]
        if n_features != len(expected_features):
            raise ValueError(
                f"Expected {len(expected_features)} weather features ({expected_features}), but received observations with {n_features} columns."
            )
    
    # Determine if request provided grid coordinates
    has_grid_coords = any(m["grid_row"] is not None and m["grid_col"] is not None for m in feature_metas)
    
    if has_grid_coords and grid_shape:
        # Build gridded tensor: [batch, seq_len, height, width, n_features]
        x_input = _build_grid_tensor(observations_list, feature_metas, grid_shape, input_steps)
    else:
        # Fall back to batch processing: [n_samples, seq_len, n_features]
        x_input = np.stack(observations_list, axis=0)
    
    # Apply scaler if available
    if bundle.scaler is not None:
        x_input = _apply_scaler(x_input, bundle.scaler)
    
    # Convert to tensor and move to device
    x_tensor = torch.from_numpy(x_input).float().to(bundle.device)
    
    # Predict
    with torch.no_grad():
        bundle.model.eval()
        
        if len(x_input.shape) == 5:
            # Gridded: [batch, seq_len, height, width, n_features]
            probs = bundle.model.predict(x_tensor).cpu().numpy()
            # probs: [batch, horizon, height, width, 1]
        else:
            # Non-gridded batch: [n_samples, seq_len, n_features]
            # Reshape to add spatial dimensionss [n_samples, seq_len, height, width, n_features]
            x_tensor = x_tensor.unsqueeze(2).unsqueeze(3)
            probs = bundle.model.predict(x_tensor).cpu().numpy()
            # probs: [n_samples, horizon, 1, 1, 1]
            probs = probs.squeeze(axis=(2, 3))  # [n_samples, horizon, 1]
    
    # Postprocess to GeoJSON
    fire_threshold = bundle.metadata.get("fire_threshold", 0.5)
    output_features = _postprocess_forecasts(
        probs, feature_metas, has_grid_coords, grid_shape, horizon, bundle.model_id, fire_threshold
    )
    
    response = ForecastResponse(
        type="FeatureCollection",
        features=output_features,
    )
    return response.model_dump(mode="json")


def _build_grid_tensor(
    observations_list: list[np.ndarray],
    feature_metas: list[dict],
    grid_shape: Tuple[int, int],
    input_steps: int,
) -> np.ndarray:
    """
    Build a gridded tensor from observations indexed by grid_row/grid_col.
    
    Args:
        observations_list: List of [seq_len, n_features] arrays
        feature_metas: List of dicts with grid_row, grid_col
        grid_shape: (height, width)
        input_steps: Sequence length
    
    Returns:
        [batch=1, seq_len, height, width, n_features] tensor
    """
    height, width = grid_shape
    n_features = observations_list[0].shape[1]
    
    # Initialize grid with NaNs
    grid = np.full((input_steps, height, width, n_features), np.nan, dtype=np.float32)
    
    # Fill grid cells
    for obs, meta in zip(observations_list, feature_metas):
        row = meta.get("grid_row")
        col = meta.get("grid_col")
        if row is not None and col is not None:
            grid[:, row, col, :] = obs
    
    # Add batch dimension
    return grid[np.newaxis, ...]  # [1, seq_len, height, width, n_features]


def _apply_scaler(x: np.ndarray, scaler: Any) -> np.ndarray:
    """
    Apply scaler to input tensor, handling both gridded and batch shapes.
    
    Args:
        x: Input array (gridded or batch)
        scaler: Fitted sklearn scaler
    
    Returns:
        Scaled array with same shape
    """
    original_shape = x.shape
    n_features = original_shape[-1]
    
    # Flatten all dimensions except features
    x_flat = x.reshape(-1, n_features)
    
    # Apply scaler
    x_scaled = scaler.transform(x_flat)
    x_scaled[np.isnan(x_scaled)] = 0.0
    
    # Reshape back
    return x_scaled.reshape(original_shape)


def _compute_risk_fields(cell_probs: list[float]) -> dict:
    """
    Derive risk_levels, risk_labels, and risk_factor from per-horizon-step
    fire probabilities. One entry per horizon step, consistent with
    fire_probability/is_burning_predicted.
    """
    risk_levels = [prob_to_risk_level(p) for p in cell_probs]
    return {
        "risk_levels": risk_levels,
        "risk_labels": [RISK_LEVEL_LABELS[lvl] for lvl in risk_levels],
        "risk_factor": [_risk_level_to_frontend_factor(lvl) for lvl in risk_levels],
    }


def _risk_level_to_frontend_factor(level: int) -> int:
    """
    Convert AI Modelling schema risk level (0=LOW..4=HIGH) to the frontend's
    risk_factor convention (1=extreme..5=very low).
    """
    return len(RISK_LEVEL_LABELS) - level


def _postprocess_forecasts(
    probs: np.ndarray,
    feature_metas: list[dict],
    has_grid_coords: bool,
    grid_shape: Optional[Tuple[int, int]],
    horizon: int,
    model_id: str,
    fire_threshold: float
) -> list[GeoFeatureOut]:
    """
    Convert fire_probability arrays back to GeoJSON features.
    
    Args:
        probs: [batch, horizon, height, width, 1] or [n_samples, horizon, 1]
        feature_metas: List of feature metadata
        has_grid_coords: Whether features have grid coordinates
        grid_shape: (height, width) if gridded
        horizon: Forecast horizon
        model_id: Model ID for response
        fire_threshold: Probability threshold for the binary prediction
    
    Returns:
        List of GeoFeatureOut
    """
    output_features = []
    
    if has_grid_coords and grid_shape and len(probs.shape) == 5:
        # Gridded forecasts: [batch, horizon, height, width, 1]
        for meta in feature_metas:
            row = meta.get("grid_row")
            col = meta.get("grid_col")
            if row is not None and col is not None:
                cell_probs = probs[0, :, row, col, 0].tolist()
                risk_score = float(np.mean(cell_probs))
                is_burning_predicted = [p > fire_threshold for p in cell_probs]
                
                props_out = ForecastPropertiesOut(
                    id=meta["id"],
                    fire_probability=cell_probs,
                    forecast=[[p] for p in cell_probs],
                    risk_score=risk_score,
                    is_burning_predicted=is_burning_predicted,
                    fire_threshold=fire_threshold,
                    horizon=horizon,
                    n_output_channels=1,
                    model_id=model_id,
                    grid_row=row,
                    grid_col=col,
                    **_compute_risk_fields(cell_probs),
                )
                feature_out = GeoFeatureOut(
                    type="Feature",
                    geometry=meta["geometry"],
                    properties=props_out.model_dump(mode="json", exclude_none=True),
                )
                output_features.append(feature_out)
    else:
        # Non-gridded batch: [n_samples, horizon, 1]
        for i, meta in enumerate(feature_metas):
            cell_probs = probs[i, :, 0].tolist()
            risk_score = float(np.mean(cell_probs))
            is_burning_predicted = [p > fire_threshold for p in cell_probs]
            
            props_out = ForecastPropertiesOut(
                id=meta["id"],
                fire_probability=cell_probs,
                forecast=[[p] for p in cell_probs],
                risk_score=risk_score,
                is_burning_predicted=is_burning_predicted,
                fire_threshold=fire_threshold,
                horizon=horizon,
                n_output_channels=1,
                model_id=model_id,
                grid_row=meta.get("grid_row"),
                grid_col=meta.get("grid_col"),
                **_compute_risk_fields(cell_probs),
            )
            feature_out = GeoFeatureOut(
                type="Feature",
                geometry=meta["geometry"],
                properties=props_out.model_dump(mode="json", exclude_none=True),
            )
            output_features.append(feature_out)
    
    return output_features