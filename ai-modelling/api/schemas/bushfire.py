"""
Bushfire I/O schemas (GeoJSON) for the single-model ConvLSTM fire predictor.

The ConvLSTM is spatiotemporal and operates on 5-D tensors end to end:

    input   [batch, seq_len, height, width, n_features]
    output  [batch, horizon, height, width, n_output_channels]

INPUT — GeoJSON FeatureCollection, one Feature per grid cell
    properties.observations : [seq_len, n_features] float matrix, channel order given by
                              ``feature_names`` (falls back to DEFAULT_FEATURE_NAMES)
    properties.grid_row/col : grid indices of the cell. Required to reassemble the real
                              5-D grid, so neighbouring cells inform each other. Without
                              them the adapter can only build a degenerate 1x1 grid
                              ([n_samples, seq_len, 1, 1, n_features]) and the spatial
                              context the ConvLSTM was trained on is lost.

Data variables (default channel order, 7 channels, ERA5-Land):
    0 skin_temperature_c                      °C
    1 soil_temperature_level_1_c              °C
    2 surface_solar_radiation_downwards       J/m^2
    3 surface_thermal_radiation_downwards     J/m^2
    4 temperature_2m_c                        °C
    5 u_component_of_wind_10m                 m/s (eastward)
    6 v_component_of_wind_10m                 m/s (northward)

OUTPUT — GeoJSON FeatureCollection, one Feature per input cell
    fire_probability     : [horizon] fire-occurrence probability in [0, 1]
    is_burning_predicted : [horizon] fire_probability > fire_threshold
    risk_score           : mean fire_probability across the horizon
    risk_levels          : [horizon] discrete level 0..4 (see RISK_LEVEL_THRESHOLDS)
    risk_labels           : [horizon] label per risk level (see RISK_LEVEL_LABELS)
    risk_factor           : [horizon] frontend risk_factor convention
                           (1=extreme..5=very low). Inverse of risk_levels
    forecast             : [horizon, n_output_channels] raw model output, one row per
                           horizon step. For the fire-probability model
                           n_output_channels == 1, so a row is [p]

Keep DEFAULT_FEATURE_NAMES, DEFAULT_INPUT_STEPS and DEFAULT_HORIZON in sync with
``src/training/ts_convlstm_forecaster_train.py``.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import Literal

DEFAULT_FEATURE_NAMES = [
    "skin_temperature_c",
    "soil_temperature_level_1_c",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
    "temperature_2m_c",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
]

# Tensor dimensions the checkpoints were trained with. The adapters prefer the values
# stored in the model bundle metadata; these are the fallbacks used when a checkpoint
# ships without them.
N_DEFAULT_FEATURES = len(DEFAULT_FEATURE_NAMES)
DEFAULT_INPUT_STEPS = 60
DEFAULT_HORIZON = 1

DEFAULT_FIRE_THRESHOLD = 0.5

RISK_LEVEL_THRESHOLDS = (0.2, 0.4, 0.6, 0.8)
RISK_LEVEL_LABELS = ("LOW", "MEDIUM_LOW", "MEDIUM", "MEDIUM_HIGH", "HIGH")


def prob_to_risk_level(prob: float) -> int:
    """Map a fire probability in [0, 1] to a discrete risk level 0..4."""
    for level, threshold in enumerate(RISK_LEVEL_THRESHOLDS):
        if prob < threshold:
            return level
    return len(RISK_LEVEL_THRESHOLDS)


def risk_level_label(level: int) -> str:
    """Human-readable label for a discrete risk level."""
    if 0 <= level < len(RISK_LEVEL_LABELS):
        return RISK_LEVEL_LABELS[level]
    return "UNKNOWN"


# ---- Input schemas ----
class FeatureTimeseriesPropertiesIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Optional[str] = None
    # observations: list of timesteps; each timestep is list of feature values in model order
    observations: List[List[float]] = Field(..., description="[[f1...fN], [f1...fN], ...] (seq_len × n_features)")
    timestamps: Optional[List[datetime]] = Field(None, description="ISO8601 timestamps aligned with observations")
    
    grid_row: Optional[int] = Field(None, ge=0, description="Row index (height axis) in the model grid")
    grid_col: Optional[int] = Field(None, ge=0, description="Column index (width axis) in the model grid")

    @field_validator("observations")
    def not_empty(cls, v):
        if not v:
            raise ValueError("observations must be a non-empty list of timesteps")
        return v

    @model_validator(mode="after")
    def validate_grid_position(self):
        if (self.grid_row is None) != (self.grid_col is None):
            raise ValueError("grid_row and grid_col must be provided together")
        return self


class GeoFeatureIn(BaseModel):
    type: Literal["Feature"]
    geometry: Dict[str, Any]
    properties: FeatureTimeseriesPropertiesIn


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["FeatureCollection"]
    features: List[GeoFeatureIn]
    # Optional override; if omitted API will use DEFAULT_FEATURE_NAMES
    feature_names: Optional[List[str]] = None
    model_id: Optional[str] = None
    schema_version: Optional[str] = Field(None, description="Payload schema version, for compatibility tracking")

    @model_validator(mode="after")
    def validate_consistent_series(self):
        if not self.features:
            raise ValueError("FeatureCollection must contain at least one Feature")
        # infer expected seq_len and n_features from first feature
        first_obs = self.features[0].properties.observations
        seq_len = len(first_obs)
        n_features = len(first_obs[0])
        if self.feature_names is not None and len(self.feature_names) != n_features:
            raise ValueError("feature_names length must match number of features per timestep")
        seen_cells: set[tuple[int, int]] = set()
        for f in self.features:
            obs = f.properties.observations
            if len(obs) != seq_len:
                raise ValueError("All features must have the same sequence length (seq_len)")
            for row in obs:
                if len(row) != n_features:
                    raise ValueError("All observation rows must have the same number of feature values")
            if f.properties.timestamps is not None and len(f.properties.timestamps) != seq_len:
                raise ValueError("timestamps (if present) must have same length as observations")
            row_idx, col_idx = f.properties.grid_row, f.properties.grid_col
            if row_idx is not None and col_idx is not None:
                cell = (row_idx, col_idx)
                if cell in seen_cells:
                    raise ValueError(f"duplicate grid cell (grid_row={row_idx}, grid_col={col_idx})")
                seen_cells.add(cell)
        return self


# ---- Output schemas ----
class ForecastPropertiesOut(BaseModel):
    """Fire prediction for one grid cell, one value per horizon step."""

    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    fire_probability: Optional[List[float]] = Field(
        None, description="Fire-occurrence probability in [0, 1], one per horizon step"
    )
    is_burning_predicted: Optional[List[bool]] = Field(
        None, description="fire_probability > fire_threshold, one per horizon step"
    )
    fire_threshold: Optional[float] = Field(None, description="Threshold used for is_burning_predicted")
    risk_score: Optional[float] = Field(None, description="Mean fire_probability across the horizon")
    risk_levels: Optional[List[int]] = Field(None, description="Discrete risk level 0..4 per horizon step")
    risk_labels: Optional[List[str]] = Field(None, description="Label per risk level (see RISK_LEVEL_LABELS)")
    risk_factor: Optional[List[int]] = Field(None, description="Frontend risk_factor convention (1=extreme..5=very low), one per horizon step. Inverse of risk_levels.")
    
    forecast: Optional[List[List[float]]] = Field(
        None, description="[horizon, n_output_channels] raw model output, one row per horizon step"
    )
    forecast_timestamps: Optional[List[datetime]] = None

    horizon: Optional[int] = Field(None, description="Number of predicted timesteps returned")
    n_output_channels: Optional[int] = Field(None, description="Number of values per predicted timestep")
    
    grid_row: Optional[int] = None
    grid_col: Optional[int] = None
    model_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_aligned_vectors(self):
        """Every per-horizon-step vector must describe the same number of steps."""
        lengths = {
            name: len(value)
            for name, value in (
                ("fire_probability", self.fire_probability),
                ("is_burning_predicted", self.is_burning_predicted),
                ("risk_levels", self.risk_levels),
                ("risk_labels", self.risk_labels),
                ("forecast", self.forecast),
            )
            if value is not None
        }
        if len(set(lengths.values())) > 1:
            raise ValueError(f"per-horizon-step fields must have the same length, got {lengths}")
        return self


class GeoFeatureOut(BaseModel):
    type: Literal["Feature"]
    geometry: Dict[str, Any]
    properties: ForecastPropertiesOut


class ForecastResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["FeatureCollection"]
    features: List[GeoFeatureOut]
