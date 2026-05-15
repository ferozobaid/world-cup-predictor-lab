"""Train all configured models."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import CONFIG, PipelineConfig
from src.constants import CLASS_LABELS
from src.models.artifacts import dump_artifact
from src.models.common import align_probabilities, split_feature_frame, target_to_index
from src.models.elo_poisson import EloPoissonHybridModel
from src.models.evaluate_models import evaluate_probability_frame
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _candidate_builders(random_state: int):
    from src.models import train_catboost, train_lightgbm, train_logistic, train_random_forest, train_xgboost

    return {
        "logistic_regression": lambda: train_logistic.build_model(random_state),
        "random_forest": lambda: train_random_forest.build_model(random_state),
        "xgboost": lambda: train_xgboost.build_model(random_state),
        "lightgbm": lambda: train_lightgbm.build_model(random_state),
        "catboost": lambda: train_catboost.build_model(random_state),
        "elo_poisson": EloPoissonHybridModel,
    }


def train_models(config: PipelineConfig = CONFIG) -> dict[str, object]:
    config.ensure_directories()
    feature_path = config.processed_dir / "match_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature matrix not found: {feature_path}. Run build-features first.")
    frame = pd.read_csv(feature_path, parse_dates=["date"])
    feature_columns, train, validation, test = split_feature_frame(frame)
    y_train = target_to_index(train["target"])

    results: dict[str, object] = {"models": {}, "feature_columns": feature_columns}
    for name, builder in _candidate_builders(config.random_state).items():
        try:
            model = builder()
            x_train = train[feature_columns].copy()
            if name == "elo_poisson":
                x_train = train[feature_columns + ["home_score", "away_score"]].copy()
            model.fit(x_train, y_train)
            model_path = config.models_dir / f"{name}.joblib"
            dump_artifact({"model": model, "feature_columns": feature_columns, "classes": CLASS_LABELS}, model_path)
            logger.info("Trained %s -> %s", name, model_path)

            eval_split = validation if not validation.empty else train
            probabilities = align_probabilities(model, model.predict_proba(eval_split[feature_columns]))
            metrics = evaluate_probability_frame(eval_split["target"], probabilities)
            results["models"][name] = {
                "status": "trained",
                "model_path": str(model_path),
                "validation_metrics": metrics,
            }
        except Exception as exc:  # Optional library failures should not break the whole pipeline.
            logger.warning("Skipping %s: %s", name, exc)
            results["models"][name] = {"status": "skipped", "reason": str(exc)}

    trained = {
        name: value
        for name, value in results["models"].items()
        if value.get("status") == "trained" and "validation_metrics" in value
    }
    if trained:
        best_name = min(trained, key=lambda key: trained[key]["validation_metrics"].get("log_loss", np.inf))
        results["best_model"] = best_name
    else:
        results["best_model"] = None

    manifest_path = config.models_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
