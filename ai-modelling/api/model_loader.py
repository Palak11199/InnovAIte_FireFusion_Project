"""
Load Hugging Face checkpoints declared in ``api/config/models.yaml``.

Extend ``_load_one`` when adding new ``kind`` values (e.g. bushfire torch checkpoints).
"""
import json
import joblib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from api.schemas.bushfire import DEFAULT_FEATURE_NAMES

_API_DIR = Path(__file__).resolve().parent
_AI_MODELLING_ROOT = _API_DIR.parent

if str(_AI_MODELLING_ROOT) not in sys.path:
    sys.path.insert(0, str(_AI_MODELLING_ROOT))

from src.models.misinformation.deberta import (
    DebertaMisinfoTrainConfig,
    load_classifier_from_checkpoint,
)


@dataclass(frozen=True)
class LoadedModel:
    """Runtime bundle for one loaded checkpoint."""

    model_id: str
    domain: str
    kind: str
    tokenizer: Any
    model: Any
    device: torch.device
    max_len: int
    checkpoint_path: Path
    scaler: Any = None  # For bushfire
    metadata: dict | None = None  # For bushfire config


_REGISTRY: dict[str, LoadedModel] = {}
_LOAD_ERRORS: list[str] = []


def _resolve_checkpoint(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    p = Path(path_value)
    if p.is_absolute():
        return p
    return (_AI_MODELLING_ROOT / p).resolve()


def _load_deberta_sequence_binary(model_id: str, domain: str, ckpt: Path) -> LoadedModel:
    if not ckpt.is_dir():
        raise FileNotFoundError(f"Checkpoint not found for '{model_id}': {ckpt}")
    tokenizer, model = load_classifier_from_checkpoint(ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    meta_path = ckpt / "training_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        max_len = int(meta.get("max_len", DebertaMisinfoTrainConfig.max_len))
    else:
        max_len = DebertaMisinfoTrainConfig.max_len
    return LoadedModel(
        model_id=model_id,
        domain=domain,
        kind="deberta_sequence_binary",
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_len=max_len,
        checkpoint_path=ckpt,
    )


def _load_one(entry: dict[str, Any]) -> LoadedModel | None:
    model_id = str(entry["id"])
    domain = str(entry.get("domain", "unknown"))
    kind = str(entry.get("kind", ""))
    enabled = bool(entry.get("enabled", True))

    if not enabled:
        return None

    if kind == "placeholder":
        return None

    if kind == "deberta_sequence_binary":
        ckpt = _resolve_checkpoint(entry.get("checkpoint"))
        if ckpt is None:
            raise ValueError(f"model '{model_id}': deberta_sequence_binary requires checkpoint")
        return _load_deberta_sequence_binary(model_id, domain, ckpt)
    
    if kind == "bushfire_forecaster":
        ckpt = _resolve_checkpoint(entry.get("checkpoint"))
        if ckpt is None:
            raise ValueError(f"model '{model_id}': bushfire_forecaster requires checkpoint")
        scaler_ckpt = _resolve_checkpoint(entry.get("scaler_checkpoint"))
        return _load_bushfire_forecaster(model_id, domain, ckpt, scaler_ckpt)

    if kind == "bushfire_classifier":
        ckpt = _resolve_checkpoint(entry.get("checkpoint"))
        if ckpt is None:
            raise ValueError(f"model '{model_id}': bushfire_classifier requires checkpoint")
        scaler_ckpt = _resolve_checkpoint(entry.get("scaler_checkpoint"))
        return _load_bushfire_classifier(model_id, domain, ckpt, scaler_ckpt)

    raise ValueError(f"model '{model_id}': unknown kind '{kind}'")


def load_models(config_path: Path | None = None) -> dict[str, LoadedModel]:
    """
    Load all enabled models from YAML. Safe to call from FastAPI lifespan.
    """
    global _REGISTRY, _LOAD_ERRORS
    _REGISTRY = {}
    _LOAD_ERRORS = []

    cfg_path = config_path or (_API_DIR / "config" / "models.yaml")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "models" not in raw:
        raise ValueError(f"{cfg_path} must be a mapping with a 'models' list")

    for entry in raw["models"]:
        if not isinstance(entry, dict):
            continue
        try:
            loaded = _load_one(entry)
            if loaded is not None:
                mid = loaded.model_id
                if mid in _REGISTRY:
                    raise ValueError(f"duplicate model id: {mid}")
                _REGISTRY[mid] = loaded
        except Exception as e:  # noqa: BLE001
            _LOAD_ERRORS.append(f"{entry.get('id', '?')}: {e}")

    return _REGISTRY.copy()


def _load_bushfire_forecaster(model_id: str, domain: str, ckpt: Path, scaler_path: Path | None = None) -> LoadedModel:
    if not ckpt.is_file():
        raise FileNotFoundError(f"Forecaster checkpoint not found for '{model_id}': {ckpt}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load forecaster checkpoint
    from src.models.bushfire.ts_convlstm_forecaster import MultivariateTSForecaster
    model = MultivariateTSForecaster.load(str(ckpt), map_location=str(device))
    model.to(device)
    model.eval()
    
    # Load scaler if provided
    scaler = None
    scaler_data = None
    if scaler_path:
        scaler_path = Path(scaler_path)
        if scaler_path.is_file():
            scaler_data = joblib.load(scaler_path)
            if isinstance(scaler_data, dict):
                scaler = scaler_data.get("scaler", scaler_data)
            else:
                scaler = scaler_data
    
    # Extract metadata from scaler or set defaults
    metadata = {}
    if scaler_data and isinstance(scaler_data, dict):
        metadata = {
            "weather_features": scaler_data.get("weather_features", DEFAULT_FEATURE_NAMES),
            "input_steps": scaler_data.get("input_steps", 30),
            "horizon": scaler_data.get("horizon", 2),
            "grid_shape": scaler_data.get("grid_shape"),
            "fire_threshold": scaler_data.get("fire_threshold", 0.5),
        }
    else:
        metadata = {
            "features": DEFAULT_FEATURE_NAMES,
            "input_steps": 30,
            "horizon": 2,
            "grid_shape": None,
            "fire_threshold": 0.5,
        }
    
    return LoadedModel(
        model_id=model_id,
        domain=domain,
        kind="bushfire_forecaster",
        tokenizer=None,
        model=model,
        device=device,
        max_len=None,
        checkpoint_path=ckpt,
        scaler=scaler,
        metadata=metadata,
    )


def _load_bushfire_classifier(model_id: str, domain: str, ckpt: Path, scaler_path: Path | None = None) -> LoadedModel:
    if not ckpt.is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found for '{model_id}': {ckpt}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load PyTorch checkpoint
    checkpoint = torch.load(ckpt, map_location=str(device), weights_only=False)
    
    # Reconstruct model from config
    if "config" in checkpoint:
        from src.models.bushfire.tcn_classifier import TCNClassifier, ClassifierConfig
        config = checkpoint.get("config")
        model = TCNClassifier(config=config)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Fallback: load directly if checkpoint is just state dict
        from src.models.bushfire.tcn_classifier import TCNClassifier, ClassifierConfig
        config = ClassifierConfig()
        model = TCNClassifier(config=config)
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Load scaler if provided
    scaler = None
    if scaler_path:
        scaler_path = Path(scaler_path)
        if scaler_path.is_file():
            scaler_data = joblib.load(scaler_path)
            if isinstance(scaler_data, dict):
                scaler = scaler_data.get("scaler", scaler_data)
            else:
                scaler = scaler_data
    
    # Build metadata dict
    metadata = {
        "lookback_steps": config.lookback_steps if "config" in checkpoint else 60,
        "n_features": config.n_features if "config" in checkpoint else 7,
        "feature_names": DEFAULT_FEATURE_NAMES,
    }
    
    return LoadedModel(
        model_id=model_id,
        domain=domain,
        kind="bushfire_classifier",
        tokenizer=None,
        model=model,
        device=device,
        max_len=None,
        checkpoint_path=ckpt,
        scaler=scaler,
        metadata=metadata,
    )


def get_model(model_id: str) -> LoadedModel:
    if model_id not in _REGISTRY:
        raise KeyError(f"unknown model_id={model_id!r}; loaded: {list(_REGISTRY.keys())}")
    return _REGISTRY[model_id]


def list_models(*, domain: str | None = None) -> list[LoadedModel]:
    out = list(_REGISTRY.values())
    if domain is not None:
        out = [m for m in out if m.domain == domain]
    return out


def default_model_id_for_domain(domain: str) -> str:
    models = list_models(domain=domain)
    if not models:
        raise RuntimeError(f"no loaded models for domain={domain!r}")
    return models[0].model_id


def is_ready() -> bool:
    return bool(_REGISTRY)


def load_errors() -> list[str]:
    return list(_LOAD_ERRORS)