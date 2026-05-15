"""XGBoost training."""

from __future__ import annotations


def build_model(random_state: int = 42):
    from xgboost import XGBClassifier

    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=250,
        max_depth=3,
        learning_rate=0.045,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="mlogloss",
        random_state=random_state,
    )

