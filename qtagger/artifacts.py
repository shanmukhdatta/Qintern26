"""Locate trusted, locally deployed model artifacts and evaluation metadata."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODEL_CANDIDATES = (
    ROOT / "models" / "qtagger_v3_1.joblib",
    ROOT / "qtagger_outputs" / "malware3" / "qtagger_v3_1.joblib",
    ROOT / "qtagger_outputs" / "four_class" / "qtagger_v3_1.joblib",
)

def model_path() -> Path | None:
    override = os.getenv("QTAGGER_MODEL_PATH", "").strip()
    if override:
        p = Path(override).expanduser()
        if not p.is_absolute():
            p = ROOT / p
        return p.resolve() if p.is_file() else None
    return next((p for p in MODEL_CANDIDATES if p.is_file()), None)

def read_metrics(model: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    metrics_file = model.with_name("evaluation_metrics.json")
    if metrics_file.is_file():
        try:
            data = json.loads(metrics_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (ValueError, OSError):
            pass
    result = bundle.get("evaluation", {})
    return result if isinstance(result, dict) else {}

def sample_csv_path(model: Path) -> Path | None:
    paths = (model.with_name("demo_unlabeled_features.csv"),
             ROOT / "models" / "demo_unlabeled_features.csv")
    return next((p for p in paths if p.is_file()), None)
