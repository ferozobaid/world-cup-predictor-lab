"""SHAP explainability exports."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.config import CONFIG, PipelineConfig
from src.constants import CLASS_LABELS, EXPORT_SCHEMA_VERSION, MODEL_VERSION
from src.exports.schemas import FrontendExport
from src.models.artifacts import load_artifact
from src.models.common import split_feature_frame
from src.utils.logging_utils import get_logger
from datetime import datetime, timezone

logger = get_logger(__name__)


def export_shap_explanations(config: PipelineConfig = CONFIG, max_background: int = 200, max_local: int = 25) -> dict[str, Any]:
    config.ensure_directories()
    output_path = config.predictions_dir / "shap_explanations.json"
    try:
        import shap
    except Exception as exc:
        data = {"status": "skipped", "reason": f"SHAP unavailable: {exc}"}
        _write(output_path, "shap_explanations.json", data)
        return data

    manifest_path = config.models_dir / "training_manifest.json"
    feature_path = config.processed_dir / "match_features.csv"
    if not manifest_path.exists() or not feature_path.exists():
        data = {"status": "skipped", "reason": "Model manifest or feature matrix missing."}
        _write(output_path, "shap_explanations.json", data)
        return data

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_name = manifest.get("best_model")
    model_info = manifest.get("models", {}).get(model_name, {})
    if not model_name or model_info.get("status") != "trained":
        data = {"status": "skipped", "reason": "No trained best model available."}
        _write(output_path, "shap_explanations.json", data)
        return data

    frame = pd.read_csv(feature_path, parse_dates=["date"])
    feature_columns, train, validation, test = split_feature_frame(frame)
    explain_frame = test if not test.empty else validation if not validation.empty else train
    background = train[feature_columns].tail(max_background)
    local_rows = explain_frame.head(max_local)
    artifact = load_artifact(model_info["model_path"])
    model = artifact["model"]

    try:
        shap_values = _compute_shap_values(shap, model, background, local_rows[feature_columns])
        global_importance = _global_importance(shap_values, feature_columns)
        local_explanations = _local_explanations(shap_values, local_rows, feature_columns)
        data = {
            "status": "complete",
            "model": model_name,
            "class_labels": CLASS_LABELS,
            "global_feature_importance": global_importance,
            "local_match_explanations": local_explanations,
        }
    except Exception as exc:
        logger.warning("SHAP failed for %s: %s", model_name, exc)
        data = {"status": "skipped", "model": model_name, "reason": str(exc)}

    _write(output_path, "shap_explanations.json", data, {"model_manifest": str(manifest_path)})
    return data


def _compute_shap_values(shap, model, background: pd.DataFrame, local: pd.DataFrame):
    estimator = model.named_steps["model"] if hasattr(model, "named_steps") and "model" in model.named_steps else model
    transformed_background = model.named_steps["scaler"].transform(background) if hasattr(model, "named_steps") and "scaler" in model.named_steps else background
    transformed_local = model.named_steps["scaler"].transform(local) if hasattr(model, "named_steps") and "scaler" in model.named_steps else local

    if estimator.__class__.__name__.lower().startswith(("randomforest", "xgb", "lgbm", "catboost")):
        explainer = shap.TreeExplainer(estimator)
        return explainer.shap_values(transformed_local)

    explainer = shap.Explainer(model.predict_proba, background)
    explanation = explainer(local)
    return explanation.values


def _as_class_array(shap_values) -> np.ndarray:
    values = np.asarray(shap_values)
    if isinstance(shap_values, list):
        values = np.stack(shap_values, axis=-1)
    if values.ndim == 2:
        values = values[:, :, np.newaxis]
    if values.shape[1] == len(CLASS_LABELS) and values.shape[2] != len(CLASS_LABELS):
        values = np.swapaxes(values, 1, 2)
    return values


def _global_importance(shap_values, feature_columns: list[str]) -> list[dict[str, Any]]:
    values = _as_class_array(shap_values)
    importance = np.mean(np.abs(values), axis=(0, 2))
    ranking = sorted(zip(feature_columns, importance), key=lambda item: float(item[1]), reverse=True)
    return [{"feature": feature, "mean_abs_shap": float(value)} for feature, value in ranking[:50]]


def _local_explanations(shap_values, rows: pd.DataFrame, feature_columns: list[str]) -> list[dict[str, Any]]:
    values = _as_class_array(shap_values)
    explanations = []
    for row_idx, (_, row) in enumerate(rows.iterrows()):
        class_contrib = values[row_idx]
        aggregate = np.mean(np.abs(class_contrib), axis=1)
        top_indices = np.argsort(aggregate)[::-1][:10]
        explanations.append(
            {
                "match_id": row["match_id"],
                "date": str(pd.Timestamp(row["date"]).date()),
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "top_features": [
                    {
                        "feature": feature_columns[idx],
                        "mean_abs_shap": float(aggregate[idx]),
                        "class_contributions": {
                            label: float(class_contrib[idx, class_idx])
                            for class_idx, label in enumerate(CLASS_LABELS)
                            if class_idx < class_contrib.shape[1]
                        },
                    }
                    for idx in top_indices
                ],
            }
        )
    return explanations


def _write(path, artifact: str, data: Any, source: dict[str, Any] | None = None) -> None:
    export = FrontendExport(
        metadata={
            "schema_version": EXPORT_SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "artifact": artifact,
            "source": source or {},
        },
        data=data,
    )
    payload = export.model_dump(mode="json") if hasattr(export, "model_dump") else export.dict()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
