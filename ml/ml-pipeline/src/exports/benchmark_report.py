"""Final model benchmark report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import CONFIG, PipelineConfig


def generate_benchmark_report(config: PipelineConfig = CONFIG) -> Path:
    evaluation_path = config.metrics_dir / "evaluation.json"
    manifest_path = config.models_dir / "training_manifest.json"
    output_path = config.metrics_dir / "benchmark_report.md"

    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8")) if evaluation_path.exists() else {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    models = evaluation.get("models", {})
    rows = []
    for name, report in models.items():
        if not isinstance(report, dict) or "log_loss" not in report:
            rows.append([name, report.get("status", "skipped"), "", "", "", report.get("reason", "") if isinstance(report, dict) else ""])
            continue
        rows.append(
            [
                name,
                "trained",
                _fmt(report.get("log_loss")),
                _fmt(report.get("macro_f1")),
                _fmt(report.get("brier_score")),
                _fmt(_calibration_gap(report)),
            ]
        )

    selected_model = evaluation.get("best_model")
    best_log_loss = _best_by_metric(models, "log_loss", lower_is_better=True)
    best_macro_f1 = _best_by_metric(models, "macro_f1", lower_is_better=False)
    best_brier = _best_by_metric(models, "brier_score", lower_is_better=True)
    best_calibrated = _best_calibrated(models)
    recommended = best_calibrated or best_log_loss or selected_model or "unavailable"

    content = [
        "# Benchmark Report",
        "",
        "## Model Comparison",
        "",
        "| Model | Status | Log Loss | Macro F1 | Brier Score | Calibration Gap |",
        "|---|---:|---:|---:|---:|---:|",
        *[f"| {name} | {status} | {log_loss} | {macro_f1} | {brier} | {calibration} |" for name, status, log_loss, macro_f1, brier, calibration in rows],
        "",
        "## Existing Baselines",
        "",
        "| Baseline | Status | Notes |",
        "|---|---|---|",
        "| Historical heuristic baseline | unavailable | Existing frontend artifacts are not present in this repo, so no direct metric import was possible. |",
        "| Existing Logistic model | unavailable | Existing frontend/model outputs are outside this standalone pipeline repo. |",
        "| Existing Elo + Score model | unavailable | Existing frontend/model outputs are outside this standalone pipeline repo. |",
        "",
        "## Recommendations",
        "",
        f"- Model selected by validation during training: `{selected_model or 'unavailable'}`.",
        f"- Best test log loss: `{best_log_loss or 'unavailable'}`.",
        f"- Best test macro F1: `{best_macro_f1 or 'unavailable'}`.",
        f"- Best test Brier score: `{best_brier or 'unavailable'}`.",
        f"- Best-calibrated model by average calibration gap: `{best_calibrated or 'unavailable'}`.",
        f"- Recommended production candidate: `{recommended}` until legacy frontend baselines are imported for direct comparison.",
        "- Preserve the architecture: offline Python training -> static JSON exports -> Next.js static consumption.",
        "- Do not run Python or train models in the frontend runtime.",
        "",
        "## Weaknesses",
        "",
        "- External legacy model metrics were not available in this workspace.",
        "- Optional enrichment datasets are adapter-ready but absent unless local CSVs are supplied.",
        "- 2026 tournament simulation depends on fixture completeness and confirmed team assignments.",
        "- Match-level features remain proxy-based because player availability, injuries, and squad market values are not yet integrated.",
        "",
        "## Frontend Integration",
        "",
        "Copy or expose the JSON files from `ml-pipeline/outputs/predictions/` into the existing Next.js static data location.",
        "The frontend should import/fetch those JSON files only; it should not call Python or train models at runtime.",
    ]
    output_path.write_text("\n".join(content) + "\n", encoding="utf-8")
    return output_path


def _fmt(value: Any) -> str:
    return "" if value is None else f"{float(value):.4f}"


def _calibration_gap(report: dict[str, Any]) -> float | None:
    curves = report.get("calibration_curve")
    if not curves:
        return None
    gaps = []
    for class_curve in curves:
        for row in class_curve.get("bins", []):
            if row.get("mean_predicted") is not None and row.get("observed_rate") is not None and row.get("count", 0) > 0:
                gaps.append(abs(row["mean_predicted"] - row["observed_rate"]))
    return sum(gaps) / len(gaps) if gaps else None


def _best_calibrated(models: dict[str, Any]) -> str | None:
    candidates = []
    for name, report in models.items():
        if isinstance(report, dict) and "log_loss" in report:
            gap = _calibration_gap(report)
            if gap is not None:
                candidates.append((gap, name))
    return min(candidates)[1] if candidates else None


def _best_by_metric(models: dict[str, Any], metric: str, lower_is_better: bool) -> str | None:
    candidates = []
    for name, report in models.items():
        if isinstance(report, dict) and metric in report:
            value = float(report[metric])
            candidates.append((value, name))
    if not candidates:
        return None
    return (min if lower_is_better else max)(candidates)[1]
