"""Model evaluation and calibration reports."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.calibration.isotonic import IsotonicProbabilityCalibrator
from src.calibration.platt_scaling import PlattProbabilityCalibrator
from src.config import CONFIG, PipelineConfig
from src.constants import CLASS_LABELS
from src.models.artifacts import load_artifact
from src.models.common import align_probabilities, split_feature_frame, target_to_index
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def evaluate_probability_frame(y_true_labels, probabilities) -> dict[str, object]:
    y_true = target_to_index(y_true_labels)
    probabilities = np.asarray(probabilities)
    predictions = probabilities.argmax(axis=1)
    y_binary = np.eye(len(CLASS_LABELS))[y_true]
    brier = float(np.mean(np.sum((probabilities - y_binary) ** 2, axis=1)))
    confusion = _confusion_matrix(y_true, predictions)
    precision, recall, f1 = _macro_precision_recall_f1(confusion)
    return {
        "accuracy": float(np.mean(y_true == predictions)),
        "precision_macro": precision,
        "recall_macro": recall,
        "macro_f1": f1,
        "log_loss": _log_loss(y_true, probabilities),
        "brier_score": brier,
        "confusion_matrix": confusion.tolist(),
    }


def _confusion_matrix(y_true: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(CLASS_LABELS), len(CLASS_LABELS)), dtype=int)
    for actual, predicted in zip(y_true, predictions):
        matrix[int(actual), int(predicted)] += 1
    return matrix


def _macro_precision_recall_f1(confusion: np.ndarray) -> tuple[float, float, float]:
    precisions = []
    recalls = []
    f1_scores = []
    for idx in range(len(CLASS_LABELS)):
        tp = confusion[idx, idx]
        fp = confusion[:, idx].sum() - tp
        fn = confusion[idx, :].sum() - tp
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1_scores))


def _log_loss(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
    return float(-np.mean(np.log(clipped[np.arange(len(y_true)), y_true])))


def calibration_curve_payload(y_true_labels, probabilities, bins: int = 10) -> list[dict[str, object]]:
    y_true = target_to_index(y_true_labels)
    payload = []
    for class_idx, label in enumerate(CLASS_LABELS):
        class_probs = np.asarray(probabilities)[:, class_idx]
        actual = (y_true == class_idx).astype(float)
        rows = []
        for low, high in zip(np.linspace(0, 1, bins, endpoint=False), np.linspace(0.1, 1, bins)):
            mask = (class_probs >= low) & (class_probs < high if high < 1 else class_probs <= high)
            rows.append(
                {
                    "bin_start": float(low),
                    "bin_end": float(high),
                    "mean_predicted": float(class_probs[mask].mean()) if mask.any() else None,
                    "observed_rate": float(actual[mask].mean()) if mask.any() else None,
                    "count": int(mask.sum()),
                }
            )
        payload.append({"class": label, "bins": rows})
    return payload


def _feature_importance(model, feature_columns: list[str]) -> list[dict[str, object]]:
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model", model)
    values = getattr(estimator, "feature_importances_", None)
    if values is None:
        coef = getattr(estimator, "coef_", None)
        if coef is not None:
            values = np.mean(np.abs(coef), axis=0)
    if values is None:
        return []
    ranking = sorted(zip(feature_columns, values), key=lambda item: abs(float(item[1])), reverse=True)
    return [{"feature": feature, "importance": float(value)} for feature, value in ranking]


def evaluate_models(config: PipelineConfig = CONFIG) -> dict[str, object]:
    config.ensure_directories()
    frame = pd.read_csv(config.processed_dir / "match_features.csv", parse_dates=["date"])
    feature_columns, _, validation, test = split_feature_frame(frame)
    eval_frame = test if not test.empty else validation
    if eval_frame.empty:
        raise ValueError("No validation or test rows available for evaluation.")

    manifest = json.loads((config.models_dir / "training_manifest.json").read_text(encoding="utf-8"))
    reports = {}
    feature_importance = {}
    prediction_examples = []

    for name, info in manifest["models"].items():
        if info.get("status") != "trained":
            reports[name] = info
            continue
        artifact = load_artifact(info["model_path"])
        model = artifact["model"]
        probabilities = align_probabilities(model, model.predict_proba(eval_frame[feature_columns]))
        report = evaluate_probability_frame(eval_frame["target"], probabilities)
        report["calibration_curve"] = calibration_curve_payload(eval_frame["target"], probabilities)
        reports[name] = report
        feature_importance[name] = _feature_importance(model, feature_columns)

        if name == manifest.get("best_model"):
            for _, row in eval_frame.head(25).iterrows():
                probs = probabilities[len(prediction_examples)]
                prediction_examples.append(
                    {
                        "match_id": row["match_id"],
                        "date": str(row["date"].date()),
                        "home_team": row["home_team"],
                        "away_team": row["away_team"],
                        "actual": row["target"],
                        "probabilities": dict(zip(CLASS_LABELS, map(float, probs))),
                    }
                )

        if not validation.empty:
            y_validation = target_to_index(validation["target"])
            validation_probs = align_probabilities(model, model.predict_proba(validation[feature_columns]))
            y_binary = np.eye(len(CLASS_LABELS))[y_validation]
            try:
                iso = IsotonicProbabilityCalibrator().fit(validation_probs, y_binary)
                reports[name]["isotonic_on_eval"] = evaluate_probability_frame(
                    eval_frame["target"], iso.predict_proba(probabilities)
                )
                platt = PlattProbabilityCalibrator(config.random_state).fit(validation_probs, y_validation)
                reports[name]["platt_on_eval"] = evaluate_probability_frame(
                    eval_frame["target"], platt.predict_proba(probabilities)
                )
            except Exception as exc:
                reports[name]["calibration_warning"] = str(exc)

    outputs = {
        "best_model": manifest.get("best_model"),
        "models": reports,
        "feature_importance": feature_importance,
        "prediction_examples": prediction_examples,
    }
    (config.metrics_dir / "evaluation.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    logger.info("Wrote evaluation report")
    return outputs
