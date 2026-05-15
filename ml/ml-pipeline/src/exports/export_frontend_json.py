"""Write stable static JSON artifacts for the frontend."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import CONFIG, PipelineConfig
from src.constants import EXPORT_SCHEMA_VERSION, MODEL_VERSION
from src.exports.benchmark_report import generate_benchmark_report
from src.exports.export_charts import generate_charts
from src.exports.export_shap import export_shap_explanations
from src.exports.schemas import FrontendExport
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _metadata(artifact: str, source: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact": artifact,
        "source": source or {},
    }


def _write(path: Path, artifact: str, data: Any, source: dict[str, Any] | None = None) -> None:
    export = FrontendExport(metadata=_metadata(artifact, source), data=data)
    payload = export.model_dump(mode="json") if hasattr(export, "model_dump") else export.dict()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", path)


def export_frontend_json(config: PipelineConfig = CONFIG) -> dict[str, str]:
    config.ensure_directories()
    evaluation_path = config.metrics_dir / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8")) if evaluation_path.exists() else {}

    exported: dict[str, str] = {}
    metrics = {"best_model": evaluation.get("best_model"), "models": evaluation.get("models", {})}
    output_specs = {
        "model_metrics.json": metrics,
        "feature_importance.json": evaluation.get("feature_importance", {}),
        "prediction_examples.json": evaluation.get("prediction_examples", []),
    }

    for filename, data in output_specs.items():
        path = config.predictions_dir / filename
        _write(path, filename, data, {"evaluation": str(evaluation_path)})
        exported[filename] = str(path)

    for filename, source_path in {
        "team_strengths.json": config.processed_dir / "team_strengths.csv",
        "elo_history.json": config.processed_dir / "elo_history.csv",
    }.items():
        data = pd.read_csv(source_path).to_dict(orient="records") if source_path.exists() else []
        path = config.predictions_dir / filename
        _write(path, filename, data, {"source_file": str(source_path)})
        exported[filename] = str(path)

    calibration = {}
    for model_name, report in evaluation.get("models", {}).items():
        if isinstance(report, dict) and "calibration_curve" in report:
            calibration[model_name] = report["calibration_curve"]
    path = config.predictions_dir / "probability_calibration.json"
    _write(path, "probability_calibration.json", calibration, {"evaluation": str(evaluation_path)})
    exported["probability_calibration.json"] = str(path)
    export_shap_explanations(config)
    exported["shap_explanations.json"] = str(config.predictions_dir / "shap_explanations.json")
    exported.update(generate_charts(config))
    exported["benchmark_report.md"] = str(generate_benchmark_report(config))
    return exported
