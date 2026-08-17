"""
Training script for the ConvLSTM bushfire classifier.

Loads two aligned gridded arrays:
  - weather/environmental features [n_timesteps, height, width, n_features]
  - binary is_burning labels       [n_timesteps, height, width, 1]

The model takes a sequence of weather grids as input and predicts the
probability that each cell is burning at the next timestep.
"""

import os
import joblib
import copy
import numpy as np
import pandas as pd
import torch
import json
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from ..models.bushfire.ts_convlstm_forecaster import ForecasterConfig, MultivariateTSForecaster
from ..evaluation.metrics import compute_metrics, find_best_threshold, format_metrics, flatten_valid

# Paths
DATA_PATH = "src/data/bushfire/forecaster_test_data.csv"
MODEL_SAVE_PATH = "src/models/bushfire/checkpoints/convlstm_forecaster.pth"
SCALER_SAVE_PATH = "src/models/bushfire/checkpoints/convlstm_scaler.pkl"

LABEL_PATH = "src/data/bushfire/historic_fire/unified_fire_data/satellite_detections_within_fires.csv"
LABEL_CACHE = "src/data/bushfire/label_grid_cache.npy"

# Model hyperparameters
INPUT_STEPS = 30
HORIZON = 1
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.001

TRAIN_VAL_RATIO = 0.9
FIRE_THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environmental features
FEATURES = [
    "era5land_temperature_2m_c",
    "era5_dewpoint_temperature_2m_c",
    "era5_total_precipitation",
    "era5_u_component_of_wind_10m",
    "era5_v_component_of_wind_10m",
    "era5land_surface_solar_radiation_downwards",
    "era5land_skin_temperature_c",
]

class MaskedTverskyLoss(nn.Module):
    """
    Tversky loss for binary bushfire classification using a spatial mask.

    Converts raw logits to probabilities and applies a spatial mask so that
    only relevant Victorian cells contribute to the loss, while ocean and
    NSW cells are excluded.

    Inputs:
        valid_mask (Tensor): Boolean [H, W] tensor — True where cells should contribute to the loss.
        alpha (float): Weight applied to false positives.
        beta (float): Weight applied to false negatives.
        smooth (float): Small value added for numerical stability.
    """

    def __init__(
        self,
        valid_mask: torch.Tensor,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-6
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

        self.register_buffer(
            "mask",
            valid_mask.float().unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Inputs:
            pred   (Tensor): [B, horizon, H, W, F] raw logits
            target (Tensor): [B, horizon, H, W, F] binary targets

        Outputs:
            Tensor: scalar masked Tversky loss
        """

        prob = torch.sigmoid(pred)

        true_positive = (
            self.mask * prob * target
        ).sum()

        false_positive = (
            self.mask * prob * (1 - target)
        ).sum()

        false_negative = (
            self.mask * (1 - prob) * target
        ).sum()

        tversky_index = (
            true_positive + self.smooth
        ) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.smooth
        )

        return 1 - tversky_index 

class MaskedFocalLoss(nn.Module):
    """
    Binary focal loss over valid (land) cells only, computed from raw logits.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma down-weights easy, well-classified examples (the vast majority of
    "no fire" cells) so the loss focuses on hard/rare positives. alpha applies
    a fixed class weight on top of that, analogous to pos_weight in BCE.

    Inputs:
        valid_mask (Tensor): Boolean [H, W] tensor — True where cells are valid.
        alpha (float): weight for the positive class, in [0, 1]. The negative class gets (1 - alpha). Higher alpha = more weight on fire cells.
        gamma (float): focusing parameter. 0 reduces to weighted BCE; typical values are 1-5. Higher gamma = more focus on hard examples.
    """
    def __init__(self, valid_mask: torch.Tensor, alpha: float = 0.85, gamma: float = 2.0) -> None:
        super().__init__()
        self.register_buffer(
            'mask',
            valid_mask.float().unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        )
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            pred (Tensor): [B, horizon, H, W, F] raw logits
            target (Tensor): [B, horizon, H, W, F] binary labels

        Outputs:
            Tensor: scalar masked focal loss
        """
        bce = self.bce(pred, target)

        p = torch.sigmoid(pred)
        p_t = p * target + (1 - p) * (1 - target)

        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = focal_weight * bce
        masked = loss * self.mask

        n_valid = self.mask.sum() * pred.shape[0] * pred.shape[1] * pred.shape[-1]
        return masked.sum() / n_valid

class GriddedTimeSeriesDataset(Dataset):
    """
    Dataset that generates sequences on the fly.

    X is drawn from the weather/environmental grid, y from the binary
    is_burning label grid, so inputs and targets are separate arrays.
    """
    def __init__(self, feature_grid, label_grid, input_steps, horizon):
        """
        Inputs:
            feature_grid (np.ndarray): [n_timesteps, height, width, n_features]
            label_grid (np.ndarray): [n_timesteps, height, width, 1]
            input_steps (int): number of input timesteps
            horizon (int): number of output timesteps
        """
        self.features = torch.tensor(feature_grid, dtype=torch.float32)
        self.labels = torch.tensor(label_grid, dtype=torch.float32)
        self.input_steps = input_steps
        self.horizon = horizon

    def __len__(self):
        return len(self.features) - self.input_steps - self.horizon + 1

    def __getitem__(self, idx):
        X = self.features[idx : idx + self.input_steps]
        y = self.labels[idx + self.input_steps : idx + self.input_steps + self.horizon]
        return X, y

def create_grid_sequences(feature_grid, label_grid, input_steps, horizon):
    """
    Create sliding-window sequences from gridded spatiotemporal data.

    Outputs:
        tuple: (X, y) where:
            - X (np.ndarray): [n_samples, input_steps, height, width, n_features]
            - y (np.ndarray): [n_samples, horizon, height, width, 1]
    """
    n_timesteps, height, width, n_features = feature_grid.shape

    max_samples = n_timesteps - input_steps - horizon + 1
    if max_samples <= 0:
        print(f"Not enough timesteps: {n_timesteps} < {input_steps + horizon}")
        return np.array([]), np.array([])

    X = np.zeros((max_samples, input_steps, height, width, n_features), dtype=np.float32)
    y = np.zeros((max_samples, horizon, height, width, 1), dtype=np.float32)

    for i in range(max_samples):
        X[i] = feature_grid[i:i + input_steps]
        y[i] = label_grid[i + input_steps:i + input_steps + horizon]

    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    return X, y

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Execute one complete training epoch on the training dataloader.
    
    Inputs:
        model (nn.Module): The neural network model to train
        dataloader (DataLoader): Training dataloader with (X, y) batches
        criterion (nn.Module): Loss function (Tversky)
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates (e.g., Adam)
        device (torch.device): Device to run training on (cuda or cpu)
    
    Outputs:
        float: Mean loss across all batches in the epoch
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        batch_size = X_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / total_samples

def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model performance on validation/test data without updating weights.
    
    Inputs:
        model (nn.Module): The neural network model to evaluate
        dataloader (DataLoader): Validation/test dataloader with (X, y) batches
        criterion (nn.Module): Loss function to compute (Tversky)
        device (torch.device): Device to run evaluation on (cuda or cpu)
    
    Outputs:
        float: Mean loss across all batches in the dataloader
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    return total_loss / total_samples

def predict(model, dataloader, device):
    """
    Generate predictions on a dataloader without computing gradients.
    
    Inputs:
        model (nn.Module): The neural network model to generate predictions
        dataloader (DataLoader): Dataloader with (X, y) batches
        device (torch.device): Device to run predictions on (cuda or cpu)
    
    Outputs:
        tuple: (predictions, actuals) where:
            - predictions (np.ndarray): Fire probabilities [n_samples, horizon, height, width, 1]
            - actuals (np.ndarray): Binary is_burning targets [n_samples, horizon, height, width, 1]
    """
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            preds = model.predict(X_batch).cpu().numpy()
            predictions.append(preds)
            actuals.append(y_batch.numpy())
    return np.concatenate(predictions), np.concatenate(actuals)

def load_and_format_gridded_data(csv_path, feature_cols=None):
    """
    Load CSV data and format it into gridded spatiotemporal format.
    
    Extracts coordinates from GeoJSON polygons, organizes data by timestamp and location,
    and returns a properly structured 4D numpy array.
    
    Inputs:
        csv_path (str): Path to the CSV file containing:
            - datetime column: Timestamp of observation
            - .geo column: GeoJSON polygon defining grid cell location
            - feature columns: Environmental measurements (7 features by default)
        feature_cols (list, optional): List of feature column names.
    
    Outputs:
        np.ndarray: Gridded data of shape [n_timesteps, height, width, n_features] with dtype=float32
    """    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV: {df.shape}")
    
    # Extract coordinates from GeoJSON
    print(f"Extracting coordinates...")
    
    def extract_coords(geojson_str):
        """
        Extract minimum longitude and latitude from GeoJSON polygon.
        
        Inputs:
            geojson_str (str): GeoJSON string representation of a polygon
        
        Outputs:
            tuple: (min_lon, min_lat) - bottom-left corner of the polygon
        """
        geojson = json.loads(geojson_str)
        coords = geojson['coordinates']
        
        # Flatten to get all individual [lon, lat] pairs
        all_lons = []
        all_lats = []
        
        # Handle nested structure: coordinates[ring][point]
        for ring in coords:
            for point in ring:
                all_lons.append(float(point[0]))
                all_lats.append(float(point[1]))
        
        return min(all_lons), min(all_lats)
    
    coords_data = []
    for idx, geojson_str in enumerate(df['.geo']):
        if idx % 100000 == 0 and idx > 0:
            print(f"Extracted {idx} coordinates...")
        
        try:
            lon, lat = extract_coords(geojson_str)
            coords_data.append({'lon': lon, 'lat': lat})
        except Exception as e:
            coords_data.append({'lon': np.nan, 'lat': np.nan})
    
    # Create DataFrame from extracted coords, merge
    coords_df = pd.DataFrame(coords_data)
    df = pd.concat([df.reset_index(drop=True), coords_df.reset_index(drop=True)], axis=1)

    # Preserve the full weather time axis before removing invalid spatial rows
    df['datetime'] = pd.to_datetime(df['datetime'])
    unique_times = sorted(df['datetime'].unique().tolist())
    print(f"Timesteps: {len(unique_times)}")

    # Remove rows where GeoJSON coordinates could not be extracted
    invalid_coords = df[['lon', 'lat']].isna().any(axis=1).sum()
    print(f"Rows with invalid coordinates removed: {invalid_coords}")
    df = df.dropna(subset=['lon', 'lat'])

    print(f"Extracted {len(coords_data) - invalid_coords} valid coordinates")

    # Get unique lat/lon values (sorted)
    unique_lats = sorted(df['lat'].unique().tolist())
    unique_lons = sorted(df['lon'].unique().tolist())
    print(f"Grid dimensions: {len(unique_lats)} x {len(unique_lons)}")

    # Create mapping
    lat_to_row = {lat: i for i, lat in enumerate(unique_lats)}
    lon_to_col = {lon: j for j, lon in enumerate(unique_lons)}
    
    # Initialize array
    n_timesteps = len(unique_times)
    height = len(unique_lats)
    width = len(unique_lons)
    n_features = len(feature_cols)

    data_grid = np.full((n_timesteps, height, width, n_features), np.nan, dtype=np.float32)
    
    print(f"Filling grid ({n_timesteps} x {height} x {width} x {n_features})...")
    # Fill the grid
    for idx, row in df.iterrows():
        if idx % 100000 == 0 and idx > 0:
            print(f"  Filled {idx} rows...")
        
        try:
            t_idx = unique_times.index(row['datetime'])
            h_idx = lat_to_row[row['lat']]
            w_idx = lon_to_col[row['lon']]
            
            data_grid[t_idx, h_idx, w_idx, :] = np.array(
                row[feature_cols].values, 
                dtype=np.float32
            )
        except (ValueError, KeyError) as e:
            continue
    
    print(f"Grid formatted: {data_grid.shape}")
    print(f"[n_timesteps={n_timesteps}, height={height}, width={width}, features={n_features}]\n")
    
    return data_grid

def load_and_format_label_grid(label_csv, weather_csv, grid_shape):
    """
    Build a binary is_burning label grid aligned to the weather data.

    Satellite detections are recorded per cell per overpass, with the 12-hour
    resolution split across the date and daynight columns. Timestamps are
    reconstructed as date + (daynight * 12h) to match the weather time axis,
    and cell_x/cell_y are used directly as grid indices since both datasets
    share the same 5km Victorian grid.

    Inputs:
        label_csv (str): Path to satellite_detections_within_fires.csv
        weather_csv (str): Path to the weather CSV, used to rebuild the time axis
        grid_shape (tuple): (height, width) of the weather grid

    Outputs:
        np.ndarray: [n_timesteps, height, width, 1] binary labels, dtype=float32
    """
    height, width = grid_shape

    # Rebuild the same time axis the weather loader used
    wt = pd.read_csv(weather_csv, usecols=['datetime'])
    wt['datetime'] = pd.to_datetime(wt['datetime'], format='mixed')
    unique_times = sorted(wt['datetime'].unique().tolist())
    time_to_idx = {t: i for i, t in enumerate(unique_times)}

    # Satellite timestamps: date + daynight (0=AM -> 00:00, 1=PM -> 12:00)
    sat = pd.read_csv(label_csv)
    sat['datetime'] = (
        pd.to_datetime(sat['datetime'])
        + pd.to_timedelta(sat['daynight'] * 12, unit='h')
    )

    label_grid = np.zeros((len(unique_times), height, width, 1), dtype=np.float32)

    placed = out_of_window = out_of_bounds = 0
    for row in sat.itertuples():
        t_idx = time_to_idx.get(row.datetime)
        if t_idx is None:
            out_of_window += 1
            continue
        if not (0 <= row.cell_y < height and 0 <= row.cell_x < width):
            out_of_bounds += 1
            continue
        label_grid[t_idx, row.cell_y, row.cell_x, 0] = 1.0
        placed += 1

    print(f"Label grid: {label_grid.shape}")
    print(f"  Placed: {placed} | outside time window: {out_of_window} "
          f"| outside grid: {out_of_bounds}")
    print(f"  Positive cells: {int(label_grid.sum())}")

    return label_grid

def compute_pos_weight(label_grid, valid_mask):
    """
    Compute the negative/positive ratio of the binary is_burning label
    over valid (land) cells only.

    Fire is extremely rare, so an unweighted loss would be minimised by
    predicting "no fire" everywhere. The returned value is intended for
    use as pos_weight in BCEWithLogitsLoss, which scales the loss
    contribution of positive cells.

    Inputs:
        label_grid (np.ndarray): [n_timesteps, height, width, 1] binary labels
        valid_mask (np.ndarray): [height, width] boolean, True where cells are valid

    Outputs:
        tuple: (pos_weight, positives, negatives, positive_rate)
    """
    valid_labels = label_grid[:, valid_mask, :]
    positives = int(valid_labels.sum())
    total = valid_labels.size
    negatives = total - positives

    if positives == 0:
        print("WARNING: no positive labels found — pos_weight defaulting to 1.0")
        return 1.0, 0, negatives, 0.0

    pos_weight = negatives / positives
    positive_rate = positives / total

    print(f"Positive cells: {positives:,} | Negative cells: {negatives:,}")
    print(f"Positive rate: {positive_rate * 100:.4f}%")
    print(f"pos_weight (neg/pos): {pos_weight:.1f}")

    return pos_weight, positives, negatives, positive_rate

def main():
    """
    Training pipeline for ConvLSTM on gridded spatiotemporal data.
    
    Workflow:
    1.  Load gridded weather features (from cache or CSV)
    1b. Load the binary is_burning label grid (from cache or CSV), aligned
        to the same time axis as the weather features
    2.  Split both features and labels into train/val/test in time order
    3.  Fit a StandardScaler on training features only (labels are never
        scaled), then scale and NaN-fill all three splits
    4.  Split train_val into train/val and build sliding-window datasets
    5.  Prepare DataLoader
    6.  Initialise the ConvLSTM model with single-channel output at horizon 1
    7.  Compute training-split class-imbalance ratio, then train
    8.  Reload the best-validation-loss model state
    9.  Select the decision threshold on validation by maximising F-beta
    10. Evaluate the model on the test set
    11. Save the trained model, scaler, and inference metadata
    """
    os.makedirs("src/models/bushfire/checkpoints", exist_ok=True)
    print("Using device:", DEVICE)
    
    print("STEP 1: Load Gridded Data")
    
    GRID_CACHE_PATH = "src/data/bushfire/data_grid_cache.npy"

    if os.path.exists(GRID_CACHE_PATH):
        print("Found cached grid, loading...")
        data_grid = np.load(GRID_CACHE_PATH)
        print(f"Loaded grid: {data_grid.shape}")
    else:
        print("No cache found, building grid from CSV...")
        data_grid = load_and_format_gridded_data(DATA_PATH, feature_cols=FEATURES)
        np.save(GRID_CACHE_PATH, data_grid)
        print(f"Grid saved to {GRID_CACHE_PATH}")

    
    n_timesteps, grid_height, grid_width, n_features = data_grid.shape
    assert n_features == len(FEATURES), f"Expected {len(FEATURES)} features, got {n_features}"

    valid_mask = ~np.all(np.isnan(data_grid), axis=(0, -1))
 
    total_cells = grid_height * grid_width
    valid_cells = valid_mask.sum()
    print(f"Valid cells: {valid_cells} / {total_cells} ({valid_cells/total_cells*100:.1f}%)")
    
    print("STEP 1b: Load Fire Label Grid")

    if os.path.exists(LABEL_CACHE):
        print("Found cached label grid, loading...")
        label_grid = np.load(LABEL_CACHE)
        print(f"Loaded label grid: {label_grid.shape}")
    else:
        print("No cache found, building label grid from CSV...")
        label_grid = load_and_format_label_grid(
            LABEL_PATH, DATA_PATH, (grid_height, grid_width)
        )
        np.save(LABEL_CACHE, label_grid)
        print(f"Label grid saved to {LABEL_CACHE}")

    assert label_grid.shape[:3] == data_grid.shape[:3], \
        f"Grid mismatch: labels {label_grid.shape} vs weather {data_grid.shape}"
 
    print("STEP 2: Split Data into Train/Val/Test")
    
    split_idx = int(len(data_grid) * TRAIN_VAL_RATIO)
    train_val_grid = data_grid[:split_idx]
    test_grid = data_grid[split_idx:]

    val_split_idx = int(len(train_val_grid) * 0.85)

    train_grid = train_val_grid[:val_split_idx]
    val_grid = train_val_grid[val_split_idx:]
    
    # Split labels on the same index so they stay aligned with the features
    train_val_labels = label_grid[:split_idx]
    test_labels      = label_grid[split_idx:]
    
    train_labels = train_val_labels[:val_split_idx]
    val_labels = train_val_labels[val_split_idx:]
    
    print(f"Train/Val: {len(train_val_grid)} timesteps")
    print(f"Test: {len(test_grid)} timesteps")
    print(f"(Split at {TRAIN_VAL_RATIO*100}% to preserve temporal order)")
    
    print("STEP 3: Fit Scaler on Training Data")
 
    # Flatten to [N, F] for sklearn on valid cells
    train_flat = train_grid.reshape(-1, n_features)

    # Keep only rows where at least one feature is not NaN - Used for evaluation
    valid_rows = ~np.all(np.isnan(train_flat), axis=1)
 
    scaler = StandardScaler()
    scaler.fit(train_flat[valid_rows])

    print(f"Scaler fitted on {valid_rows.sum()} valid cell-timesteps")
    print(f"Feature means: {scaler.mean_}")
    print(f"Feature scales: {scaler.scale_}")
 
    def scale_and_fill(grid: np.ndarray) -> np.ndarray:
        """Scale [T, H, W, F] grid and replace NaNs with 0."""
        shape = grid.shape
        flat = grid.reshape(-1, n_features)
        scaled = scaler.transform(flat)
        scaled[np.isnan(scaled)] = 0.0
        return scaled.reshape(shape)
 
    train_scaled = scale_and_fill(train_grid)
    val_scaled = scale_and_fill(val_grid)
    test_scaled = scale_and_fill(test_grid)

    print("STEP 4: Create Datasets with Sliding Window")

    # Split features and labels into train/validation sets
    train_dataset = GriddedTimeSeriesDataset(train_scaled, train_labels, INPUT_STEPS, HORIZON)
    val_dataset   = GriddedTimeSeriesDataset(val_scaled, val_labels, INPUT_STEPS, HORIZON)
    test_dataset  = GriddedTimeSeriesDataset(test_scaled, test_labels, INPUT_STEPS, HORIZON)

    print(f"Train timesteps: {len(train_scaled)}")
    print(f"Val timesteps: {len(val_scaled)}")
    print(f"Test timesteps: {len(test_scaled)}")
    print(f"Minimum needed: {INPUT_STEPS + HORIZON}")

    print(f"Train sequences: {len(train_dataset)}")
    print(f"Val sequences: {len(val_dataset)}")
    print(f"Test sequences: {len(test_dataset)}")

    print("STEP 5: Create DataLoaders")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print("STEP 6: Initialise ConvLSTM Model")
    
    config = ForecasterConfig(
        input_channels=n_features,
        horizon=HORIZON,
        output_channels=1,
        hidden_size_1=32,
        hidden_size_2=16,
        dropout=0.2
    )
    
    model = MultivariateTSForecaster(config).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("STEP 7: Train Model")
    
    valid_mask_tensor = torch.tensor(valid_mask, dtype=torch.bool)
    
    # Class imbalance ratio for the loss function (training split only, to avoid leaking val/test distribution).
    pos_weight, _, _, _ = compute_pos_weight(train_labels, valid_mask)
    alpha = pos_weight / (1 + pos_weight)
    
    criterion = MaskedTverskyLoss(valid_mask_tensor, alpha=0.3, beta=0.7).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float("inf")
    best_state = None
    patience = 10
    patience_counter = 0
    
    print(f"Training config:")
    print(f"Loss: Tversky")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Early stopping patience: {patience}")
    
    print(f"\n{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Status':<15}")
    print("-" * 60)
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = evaluate(model, val_loader, criterion, DEVICE)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            status = "BEST"
        else:
            patience_counter += 1
            status = f"Wait {patience_counter}/{patience}"
        
        print(f"{epoch:<8} {train_loss:<15.6f} {val_loss:<15.6f} {status:<15}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    print("STEP 8: Load Best Model")
    
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model (val_loss={best_val_loss:.6f})")
    else:
        print(f"Using final model")
    
    print("STEP 9: Select Threshold on Validation Set")
    
    y_val_prob, y_val_true = predict(model, val_loader, DEVICE)
    val_true_flat, val_prob_flat = flatten_valid(y_val_true, y_val_prob, valid_mask)

    best_threshold, best_val_fbeta = find_best_threshold(val_true_flat, val_prob_flat, beta=2.0)
    if best_val_fbeta is None:
        best_threshold = FIRE_THRESHOLD
        print(f"Threshold defaulted to {FIRE_THRESHOLD}")
    else:
        print(f"Selected threshold: {best_threshold:.4f} (val F2 = {best_val_fbeta:.4f})")

    print("STEP 10: Evaluate on Test Set")

    y_test_prob, y_test_true = predict(model, test_loader, DEVICE)
    test_true_flat, test_prob_flat = flatten_valid(y_test_true, y_test_prob, valid_mask)

    test_metrics = compute_metrics(test_true_flat, test_prob_flat, threshold=best_threshold, beta=2.0)
    print(format_metrics(test_metrics, title="Test Set Metrics"))
    
    print("STEP 11: Save Model and Scaler")
    
    model.save(MODEL_SAVE_PATH)
    
    joblib.dump(
        {
            "scaler": scaler,
            # Key name changed to "weather_features" as the inference module 
            # reads bundle.metadata["weather_features"] to validate the incoming feature count
            "weather_features": FEATURES,
            "input_steps": INPUT_STEPS,
            "horizon": HORIZON,
            "grid_shape": (grid_height, grid_width),
            "fire_threshold": best_threshold,
        },
        SCALER_SAVE_PATH
    )
    
    print(f"Saved model: src/models/bushfire/checkpoints/convlstm_forecaster.pth")
    print(f"Saved scaler: src/models/bushfire/checkpoints/convlstm_scaler.pkl")
    
    print("TRAINING COMPLETE")

if __name__ == "__main__":
    main()
