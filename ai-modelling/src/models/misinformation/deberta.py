"""
DeBERTa v3-large: model wiring and data objects that feed the classifier.

Checkpoints from ``build_fresh_classifier`` / ``save_pretrained`` include ``id2label``
and ``label2id`` on the model config. Training loops, metrics, and seeding live in
``src/training/deberta_train.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

DEFAULT_HF_MODEL_ID = "microsoft/deberta-v3-large"

DEFAULT_ID2LABEL = {0: "non_misinformation", 1: "misinformation"}
DEFAULT_LABEL2ID = {v: k for k, v in DEFAULT_ID2LABEL.items()}


@dataclass(frozen=True)
class DebertaMisinfoTrainConfig:
    hf_model_id: str = DEFAULT_HF_MODEL_ID
    max_len: int = 256
    num_labels: int = 2


def load_table(
    path: Path,
    text_col: str,
    label_col: str,
    *,
    valid_labels: tuple[int, ...] = (0, 1),
) -> pd.DataFrame:
    """
    Load a (text, label) table. ``valid_labels`` bounds the allowed integer labels:
    keep the default ``(0, 1)`` for binary misinfo, pass ``(0, 1, 2)`` for urgency,
    ``tuple(range(6))`` for humanitarian, etc.
    """
    suffix = path.suffix.lower()
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"{path} must contain a JSON list of records")
        df = pd.DataFrame(raw)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path} (use .json or .csv)")

    if text_col not in df.columns or label_col not in df.columns:
        raise KeyError(f"{path} must have columns {text_col!r} and {label_col!r}; got {list(df.columns)}")

    out = pd.DataFrame({"text": df[text_col].astype(str), "label": df[label_col]})
    out["label"] = pd.to_numeric(out["label"], errors="raise").astype(int)
    bad = ~out["label"].isin(list(valid_labels))
    if bool(bad.any()):
        raise ValueError(f"Labels must be in {valid_labels}; offending rows: {out.loc[bad].head()}")
    return out.reset_index(drop=True)


class TextClsDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer: Any, max_len: int):
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        if max_len <= 0:
            raise ValueError("max_len must be greater than zero")
        self.texts = texts
        self.labels = [int(x) for x in labels]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def collate_text_cls_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("batch must not be empty")
    keys = [k for k in batch[0].keys() if k != "labels"]
    out: dict[str, Any] = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["labels"] = torch.stack([b["labels"] for b in batch], dim=0)
    return out


def build_fresh_classifier(
    hf_model_id: str = DEFAULT_HF_MODEL_ID,
    *,
    num_labels: int = 2,
    id2label: dict[int, str] | None = None,
    label2id: dict[str, int] | None = None,
) -> tuple[Any, Any]:
    id2label = id2label or DEFAULT_ID2LABEL
    label2id = label2id or DEFAULT_LABEL2ID
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    return tokenizer, model


def load_classifier_from_checkpoint(checkpoint_dir: Path | str) -> tuple[Any, Any]:
    path = Path(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    return tokenizer, model


@torch.no_grad()
def classify_text(
    text: str,
    *,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    max_len: int,
) -> dict[str, Any]:
    """
    Tokenize a single string and run the classifier head (softmax probs).
    API layers can add severity, rationale, etc.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if max_len <= 0:
        raise ValueError("max_len must be greater than zero")
    model.eval()
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).squeeze(0)
    label_id = int(torch.argmax(probs).item())
    raw_map = getattr(model.config, "id2label", None) or {}
    id2label: dict[int, str] = {int(k): str(v) for k, v in dict(raw_map).items()}
    label_name = id2label.get(label_id, str(label_id))
    probabilities = {id2label.get(i, f"class_{i}"): float(probs[i].item()) for i in range(probs.shape[0])}
    return {
        "label_id": label_id,
        "label": label_name,
        "confidence": float(probs[label_id].item()),
        "probabilities": probabilities,
    }


# ---------------------------------------------------------------------------
# Multi-task extension
#
# Everything above is the original single-task (misinformation-only) path and is
# left untouched. The section below adds a multi-task model: one shared
# DeBERTa encoder with a separate classification head per task
# (misinformation + urgency + humanitarian). See
# ``notebooks/research/RESEARCH_Urgency_Classification_Integration.md``.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskSpec:
    """One classification task: its name, how many labels it has, and id -> name."""

    name: str
    num_labels: int
    id2label: dict[int, str]


# Each task is just a (name, num_labels, id2label) triple. Adding a task = add one here.
MISINFO_TASK = TaskSpec("misinfo", 2, {0: "FALSE", 1: "TRUE"})
URGENCY_TASK = TaskSpec("urgency", 3, {0: "NOT_USEFUL", 1: "NOT_URGENT", 2: "URGENT"})
HUMANITARIAN_TASK = TaskSpec(
    "humanitarian",
    6,
    {
        0: "HMN_DMG",    # human damage
        1: "MAT_DMG",    # material damage
        2: "WARN",       # warning
        3: "EVAC",       # evacuations
        4: "HMN_MISS",   # missing people
        5: "VOLUNTEER",  # volunteering
    },
)

DEFAULT_TASKS: tuple[TaskSpec, ...] = (MISINFO_TASK, URGENCY_TASK, HUMANITARIAN_TASK)


class MultiTaskDeberta(nn.Module):
    """
    Shared DeBERTa encoder + one linear classification head per task.

    - ``encode`` runs the (expensive) transformer once and pools to one vector/post.
    - ``forward(tasks=[...])`` applies only the requested heads (training uses the
      head(s) the batch has labels for; inference leaves ``tasks=None`` for all heads).
    """

    def __init__(
        self,
        hf_model_id: str = DEFAULT_HF_MODEL_ID,
        *,
        tasks: tuple[TaskSpec, ...] = DEFAULT_TASKS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hf_model_id = hf_model_id
        self.config = AutoConfig.from_pretrained(hf_model_id)
        self.encoder = AutoModel.from_pretrained(hf_model_id)
        self.dropout = nn.Dropout(dropout)

        hidden = self.config.hidden_size
        # ModuleDict registers each head as a real submodule (trained, moved, saved).
        self.heads = nn.ModuleDict(
            {t.name: nn.Linear(hidden, t.num_labels) for t in tasks}
        )
        self.tasks: dict[str, TaskSpec] = {t.name: t for t in tasks}

    def encode(self, input_ids, attention_mask, token_type_ids=None) -> torch.Tensor:
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # DeBERTa exposes no pooler under AutoModel; use the first token ([CLS]).
        return self.dropout(out.last_hidden_state[:, 0])

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        *,
        tasks: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encode(input_ids, attention_mask, token_type_ids)
        names = tasks if tasks is not None else list(self.heads.keys())
        return {name: self.heads[name](hidden) for name in names}


@torch.no_grad()
def classify_multitask(
    text: str,
    *,
    tokenizer: Any,
    model: MultiTaskDeberta,
    device: torch.device,
    max_len: int,
) -> dict[str, dict[str, Any]]:
    """
    Run every head on one string. Returns, per task, the predicted label, its
    confidence, and the full probability map:
    ``{"misinfo": {...}, "urgency": {...}, "humanitarian": {...}}``.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if max_len <= 0:
        raise ValueError("max_len must be greater than zero")

    model.eval()
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    all_logits = model(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        token_type_ids=enc.get("token_type_ids"),
    )

    result: dict[str, dict[str, Any]] = {}
    for name, logits in all_logits.items():
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        label_id = int(torch.argmax(probs).item())
        id2label = model.tasks[name].id2label
        result[name] = {
            "label_id": label_id,
            "label": id2label.get(label_id, str(label_id)),
            "confidence": float(probs[label_id].item()),
            "probabilities": {
                id2label.get(i, f"class_{i}"): float(probs[i].item())
                for i in range(probs.shape[0])
            },
        }
    return result


def save_multitask_checkpoint(
    model: MultiTaskDeberta,
    tokenizer: Any,
    output_dir: Path | str,
    *,
    max_len: int,
) -> None:
    """
    A custom module can't use ``save_pretrained``. Persist four things so the model
    can be rebuilt exactly: encoder config, tokenizer, weights, and a task manifest.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.encoder.config.save_pretrained(out)
    tokenizer.save_pretrained(out)
    torch.save(model.state_dict(), out / "model.pt")
    manifest = {
        "hf_model_id": model.hf_model_id,
        "max_len": max_len,
        "tasks": [
            {"name": t.name, "num_labels": t.num_labels, "id2label": t.id2label}
            for t in model.tasks.values()
        ],
    }
    (out / "tasks.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_multitask_from_checkpoint(
    checkpoint_dir: Path | str,
    *,
    device: torch.device | None = None,
) -> tuple[Any, MultiTaskDeberta, int]:
    """
    Rebuild the head structure from ``tasks.json`` first, then load the saved weights
    into it. If the manifest and code disagree on head shapes, ``load_state_dict``
    fails loudly instead of loading a mismatched head.
    """
    path = Path(checkpoint_dir)
    spec = json.loads((path / "tasks.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(path)
    tasks = tuple(
        TaskSpec(t["name"], int(t["num_labels"]), {int(k): str(v) for k, v in t["id2label"].items()})
        for t in spec["tasks"]
    )
    model = MultiTaskDeberta(spec["hf_model_id"], tasks=tasks)
    state = torch.load(path / "model.pt", map_location=device or "cpu")
    model.load_state_dict(state)
    if device is not None:
        model.to(device)
    return tokenizer, model, int(spec["max_len"])