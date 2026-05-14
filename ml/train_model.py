from __future__ import annotations

import json
import math
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "src" / "data" / "world-cup-data.json"
OUTPUT_DIR = ROOT / "ml" / "model_outputs"

FEATURE_COLUMNS = [
    "team_a_win_rate",
    "team_b_win_rate",
    "team_a_goals_for_per_match",
    "team_b_goals_for_per_match",
    "team_a_goals_against_per_match",
    "team_b_goals_against_per_match",
    "team_a_goal_difference_per_match",
    "team_b_goal_difference_per_match",
    "team_a_recent_form_index",
    "team_b_recent_form_index",
    "team_a_tournament_experience",
    "team_b_tournament_experience",
    "knockout_match_flag",
    "final_flag",
    "strength_difference",
    "form_difference",
    "goal_balance_difference",
    "experience_difference",
]

SCORE_FEATURE_COLUMNS = [
    "team_a_win_rate",
    "team_b_win_rate",
    "team_a_goals_for_per_match",
    "team_b_goals_for_per_match",
    "team_a_goals_against_per_match",
    "team_b_goals_against_per_match",
    "team_a_goal_difference_per_match",
    "team_b_goal_difference_per_match",
    "team_a_recent_form_index",
    "team_b_recent_form_index",
    "team_a_tournament_experience",
    "team_b_tournament_experience",
    "knockout_match_flag",
    "final_flag",
    "stage_importance",
    "host_a_flag",
    "host_b_flag",
    "neutral_site_flag",
    "elo_a",
    "elo_b",
    "elo_difference",
    "strength_difference",
    "form_difference",
    "goal_balance_difference",
    "experience_difference",
]

CLASS_LABELS = ["teamA_win", "draw", "teamB_win"]


@dataclass
class TeamHistory:
    matches: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    recent_points: deque[int] = field(default_factory=lambda: deque(maxlen=10))
    tournaments: set[int] = field(default_factory=set)

    def snapshot(self) -> dict[str, float]:
        denom = max(self.matches, 1)
        recent_denom = max(len(self.recent_points) * 3, 1)
        win_rate = self.wins / denom if self.matches else 0.33
        goals_for = self.goals_for / denom if self.matches else 1.0
        goals_against = self.goals_against / denom if self.matches else 1.0
        goal_diff = goals_for - goals_against
        recent_form = sum(self.recent_points) / recent_denom if self.recent_points else 0.33
        tournament_experience = len(self.tournaments)
        strength = win_rate * 48 + goal_diff * 10 + recent_form * 18 + math.log1p(self.matches) * 3
        return {
            "win_rate": win_rate,
            "goals_for_per_match": goals_for,
            "goals_against_per_match": goals_against,
            "goal_difference_per_match": goal_diff,
            "recent_form_index": recent_form,
            "tournament_experience": tournament_experience,
            "strength": strength,
            "matches_before": self.matches,
        }

    def update(self, year: int, goals_for: int, goals_against: int) -> None:
        self.matches += 1
        self.goals_for += goals_for
        self.goals_against += goals_against
        self.tournaments.add(year)
        if goals_for > goals_against:
            self.wins += 1
            self.recent_points.append(3)
        elif goals_for == goals_against:
            self.draws += 1
            self.recent_points.append(1)
        else:
            self.losses += 1
            self.recent_points.append(0)


def load_matches() -> list[dict]:
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return sorted(data["matches"], key=lambda match: (match["date"], match["id"]))


def target_for_match(match: dict) -> str:
    if match["homeScore"] > match["awayScore"]:
        return "teamA_win"
    if match["homeScore"] < match["awayScore"]:
        return "teamB_win"
    return "draw"


def stage_importance(stage: str) -> float:
    normalized = stage.lower()
    if normalized == "final":
        return 1.5
    if normalized == "semi-finals":
        return 1.35
    if normalized == "quarter-finals":
        return 1.25
    if normalized == "round of 16":
        return 1.15
    if normalized in {"second group stage", "final round"}:
        return 1.05
    return 0.9


def match_result_value(goals_for: int, goals_against: int) -> float:
    if goals_for > goals_against:
        return 1.0
    if goals_for == goals_against:
        return 0.5
    return 0.0


def update_elo(rating_a: float, rating_b: float, goals_a: int, goals_b: int, importance: float) -> tuple[float, float]:
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    result_a = match_result_value(goals_a, goals_b)
    margin = abs(goals_a - goals_b)
    margin_multiplier = 1.0 if margin <= 1 else math.log1p(margin)
    k_factor = 24 * importance * margin_multiplier
    delta = k_factor * (result_a - expected_a)
    return rating_a + delta, rating_b - delta


def host_flag(match: dict, team: str) -> int:
    return 1 if match["hostCountry"] == team else 0


def build_feature_rows(matches: list[dict]) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    histories: dict[str, TeamHistory] = {}
    elo_ratings: dict[str, float] = {}
    rows: list[dict] = []

    for match in matches:
        team_a = match["homeTeam"]
        team_b = match["awayTeam"]
        histories.setdefault(team_a, TeamHistory())
        histories.setdefault(team_b, TeamHistory())
        elo_ratings.setdefault(team_a, 1500.0)
        elo_ratings.setdefault(team_b, 1500.0)

        a = histories[team_a].snapshot()
        b = histories[team_b].snapshot()
        importance = stage_importance(match["stage"])
        host_a = host_flag(match, team_a)
        host_b = host_flag(match, team_b)
        row = {
            "match_id": match["id"],
            "year": match["year"],
            "date": match["date"],
            "team_a": team_a,
            "team_b": team_b,
            "stage": match["stage"],
            "target": target_for_match(match),
            "team_a_goals": match["homeScore"],
            "team_b_goals": match["awayScore"],
            "team_a_win_rate": a["win_rate"],
            "team_b_win_rate": b["win_rate"],
            "team_a_goals_for_per_match": a["goals_for_per_match"],
            "team_b_goals_for_per_match": b["goals_for_per_match"],
            "team_a_goals_against_per_match": a["goals_against_per_match"],
            "team_b_goals_against_per_match": b["goals_against_per_match"],
            "team_a_goal_difference_per_match": a["goal_difference_per_match"],
            "team_b_goal_difference_per_match": b["goal_difference_per_match"],
            "team_a_recent_form_index": a["recent_form_index"],
            "team_b_recent_form_index": b["recent_form_index"],
            "team_a_tournament_experience": a["tournament_experience"],
            "team_b_tournament_experience": b["tournament_experience"],
            "knockout_match_flag": 1 if match["knockoutStage"] else 0,
            "final_flag": 1 if match["stage"] == "final" else 0,
            "stage_importance": importance,
            "host_a_flag": host_a,
            "host_b_flag": host_b,
            "neutral_site_flag": 1 if not host_a and not host_b else 0,
            "elo_a": elo_ratings[team_a],
            "elo_b": elo_ratings[team_b],
            "elo_difference": elo_ratings[team_a] - elo_ratings[team_b],
            "strength_difference": a["strength"] - b["strength"],
            "form_difference": a["recent_form_index"] - b["recent_form_index"],
            "goal_balance_difference": a["goal_difference_per_match"] - b["goal_difference_per_match"],
            "experience_difference": a["tournament_experience"] - b["tournament_experience"],
        }
        rows.append(row)

        histories[team_a].update(match["year"], match["homeScore"], match["awayScore"])
        histories[team_b].update(match["year"], match["awayScore"], match["homeScore"])
        elo_ratings[team_a], elo_ratings[team_b] = update_elo(
            elo_ratings[team_a],
            elo_ratings[team_b],
            match["homeScore"],
            match["awayScore"],
            importance,
        )

    latest_features = {
        team: {
            **history.snapshot(),
            "elo_rating": elo_ratings.get(team, 1500.0),
        }
        for team, history in histories.items()
    }
    return pd.DataFrame(rows), latest_features


def evaluate_model(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "model_name": name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASS_LABELS).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }


def add_probability_metrics(metrics: dict, y_true: pd.Series, probabilities: np.ndarray) -> dict:
    class_index = {label: index for index, label in enumerate(CLASS_LABELS)}
    y_true_one_hot = np.zeros((len(y_true), len(CLASS_LABELS)))
    for row_index, label in enumerate(y_true):
        y_true_one_hot[row_index, class_index[label]] = 1
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    metrics["log_loss"] = round(float(log_loss(y_true, clipped, labels=CLASS_LABELS)), 4)
    metrics["brier_score"] = round(float(np.mean(np.sum((clipped - y_true_one_hot) ** 2, axis=1))), 4)
    return metrics


def align_probabilities(model_classes: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    class_to_index = {label: index for index, label in enumerate(model_classes)}
    return np.column_stack([probabilities[:, class_to_index[label]] for label in CLASS_LABELS])


def poisson_pmf(mean: float, max_goals: int = 8) -> np.ndarray:
    mean = float(np.clip(mean, 0.05, 5.0))
    values = [math.exp(-mean)]
    for goals in range(1, max_goals + 1):
        values.append(values[-1] * mean / goals)
    return np.array(values)


def score_distribution(lambda_a: float, lambda_b: float, max_goals: int = 8) -> tuple[np.ndarray, tuple[int, int]]:
    dist_a = poisson_pmf(lambda_a, max_goals)
    dist_b = poisson_pmf(lambda_b, max_goals)
    matrix = np.outer(dist_a, dist_b)
    total = matrix.sum()
    if total > 0:
        matrix = matrix / total
    team_a_win = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    team_b_win = float(np.triu(matrix, 1).sum())
    score_index = np.unravel_index(np.argmax(matrix), matrix.shape)
    return np.array([team_a_win, draw, team_b_win]), (int(score_index[0]), int(score_index[1]))


def elo_score_predictions(
    rows: pd.DataFrame,
    goal_model_a: PoissonRegressor,
    goal_model_b: PoissonRegressor,
    scaler: StandardScaler,
) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray, np.ndarray]:
    features = rows[SCORE_FEATURE_COLUMNS]
    scaled = scaler.transform(features)
    lambdas_a = np.clip(goal_model_a.predict(scaled), 0.1, 4.8)
    lambdas_b = np.clip(goal_model_b.predict(scaled), 0.1, 4.8)
    probability_rows = []
    likely_scores = []
    for lambda_a, lambda_b in zip(lambdas_a, lambdas_b):
        probabilities, score = score_distribution(lambda_a, lambda_b)
        probability_rows.append(probabilities)
        likely_scores.append(score)
    probability_matrix = np.array(probability_rows)
    return probability_matrix, likely_scores, lambdas_a, lambdas_b


def score_class_predictions(probabilities: np.ndarray, draw_margin: float) -> np.ndarray:
    predictions = []
    for team_a_win, draw, team_b_win in probabilities:
        strongest_win = max(team_a_win, team_b_win)
        if draw >= strongest_win - draw_margin:
            predictions.append("draw")
        elif team_a_win > team_b_win:
            predictions.append("teamA_win")
        else:
            predictions.append("teamB_win")
    return np.array(predictions)


def optimize_draw_margin(y_true: pd.Series, probabilities: np.ndarray) -> float:
    best_margin = 0.0
    best_score = -1.0
    for margin in np.linspace(0, 0.18, 19):
        preds = score_class_predictions(probabilities, float(margin))
        score = f1_score(y_true, preds, average="macro")
        if score > best_score:
            best_score = score
            best_margin = float(margin)
    return best_margin


def historical_strength_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, float]:
    best_threshold = 0.0
    best_score = -1.0
    for threshold in np.linspace(0, 12, 25):
        preds = strength_predictions(train_df["strength_difference"].to_numpy(), threshold)
        score = f1_score(train_df["target"], preds, average="macro")
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return strength_predictions(test_df["strength_difference"].to_numpy(), best_threshold), best_threshold


def strength_predictions(strength_diff: np.ndarray, threshold: float) -> np.ndarray:
    preds = []
    for value in strength_diff:
        if value > threshold:
            preds.append("teamA_win")
        elif value < -threshold:
            preds.append("teamB_win")
        else:
            preds.append("draw")
    return np.array(preds)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


def prediction_example(row: pd.Series, model: LogisticRegression, scaler: StandardScaler) -> dict:
    features = pd.DataFrame([row[FEATURE_COLUMNS].to_dict()], columns=FEATURE_COLUMNS)
    probabilities = model.predict_proba(scaler.transform(features))[0]
    return {
        "match_id": row["match_id"],
        "year": int(row["year"]),
        "teamA": row["team_a"],
        "teamB": row["team_b"],
        "actual": row["target"],
        "probabilities": {label: round(float(prob), 4) for label, prob in zip(model.classes_, probabilities)},
    }


def score_prediction_example(
    row: pd.Series,
    goal_model_a: PoissonRegressor,
    goal_model_b: PoissonRegressor,
    scaler: StandardScaler,
) -> dict:
    features = pd.DataFrame([row[SCORE_FEATURE_COLUMNS].to_dict()], columns=SCORE_FEATURE_COLUMNS)
    scaled = scaler.transform(features)
    lambda_a = float(np.clip(goal_model_a.predict(scaled)[0], 0.1, 4.8))
    lambda_b = float(np.clip(goal_model_b.predict(scaled)[0], 0.1, 4.8))
    probabilities, likely_score = score_distribution(lambda_a, lambda_b)
    return {
        "match_id": row["match_id"],
        "year": int(row["year"]),
        "teamA": row["team_a"],
        "teamB": row["team_b"],
        "actual": row["target"],
        "expected_goals": {
            "teamA": round(lambda_a, 3),
            "teamB": round(lambda_b, 3),
        },
        "likely_score": {
            "teamA": likely_score[0],
            "teamB": likely_score[1],
        },
        "probabilities": {
            label: round(float(prob), 4)
            for label, prob in zip(CLASS_LABELS, probabilities)
        },
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matches = load_matches()
    df, latest_team_features = build_feature_rows(matches)

    train_df = df[df["year"] < 2014].copy()
    test_df = df[df["year"] >= 2014].copy()
    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target"]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["target"]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    logistic = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    logistic.fit(x_train_scaled, y_train)
    logistic_pred = logistic.predict(x_test_scaled)

    forest = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )
    forest.fit(x_train, y_train)
    forest_pred = forest.predict(x_test)

    gradient = CalibratedClassifierCV(
        estimator=GradientBoostingClassifier(random_state=42),
        method="sigmoid",
        cv=3,
    )
    gradient.fit(x_train_scaled, y_train)
    gradient_pred = gradient.predict(x_test_scaled)

    score_scaler = StandardScaler()
    x_score_train = train_df[SCORE_FEATURE_COLUMNS]
    x_score_test = test_df[SCORE_FEATURE_COLUMNS]
    x_score_train_scaled = score_scaler.fit_transform(x_score_train)
    goal_model_a = PoissonRegressor(alpha=0.08, max_iter=1000)
    goal_model_b = PoissonRegressor(alpha=0.08, max_iter=1000)
    goal_model_a.fit(x_score_train_scaled, train_df["team_a_goals"])
    goal_model_b.fit(x_score_train_scaled, train_df["team_b_goals"])
    train_score_probs, _, _, _ = elo_score_predictions(
        train_df,
        goal_model_a,
        goal_model_b,
        score_scaler,
    )
    draw_margin = optimize_draw_margin(y_train, train_score_probs)
    elo_score_probs, _, _, _ = elo_score_predictions(
        test_df,
        goal_model_a,
        goal_model_b,
        score_scaler,
    )
    elo_score_pred = score_class_predictions(elo_score_probs, draw_margin)

    majority_class = Counter(y_train).most_common(1)[0][0]
    majority_pred = np.array([majority_class] * len(y_test))
    strength_pred, strength_threshold = historical_strength_baseline(train_df, test_df)

    logistic_probs = align_probabilities(logistic.classes_, logistic.predict_proba(x_test_scaled))
    forest_probs = align_probabilities(forest.classes_, forest.predict_proba(x_test))
    gradient_probs = align_probabilities(gradient.classes_, gradient.predict_proba(x_test_scaled))

    logistic_metrics = add_probability_metrics(evaluate_model("Logistic Regression", y_test, logistic_pred), y_test, logistic_probs)
    forest_metrics = add_probability_metrics(evaluate_model("Random Forest Classifier", y_test, forest_pred), y_test, forest_probs)
    gradient_metrics = add_probability_metrics(evaluate_model("Calibrated Gradient Boosting", y_test, gradient_pred), y_test, gradient_probs)
    elo_score_metrics = add_probability_metrics(evaluate_model("Elo + Poisson Score Model", y_test, elo_score_pred), y_test, elo_score_probs)
    majority_metrics = evaluate_model("Majority Class Baseline", y_test, majority_pred)
    strength_metrics = evaluate_model("Historical Strength Baseline", y_test, strength_pred)

    model_metrics = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "dataset": {
            "total_matches": int(len(df)),
            "train_matches": int(len(train_df)),
            "test_matches": int(len(test_df)),
            "train_years": [int(train_df["year"].min()), int(train_df["year"].max())],
            "test_years": [int(test_df["year"].min()), int(test_df["year"].max())],
            "target_classes": CLASS_LABELS,
        },
        "selected_frontend_model": "Logistic Regression",
        "probability_note": "These are model-estimated probabilities, not betting odds. Calibration may be weak because historical World Cup data is small.",
        "models": {
            "logistic_regression": logistic_metrics,
            "random_forest": forest_metrics,
            "calibrated_gradient_boosting": gradient_metrics,
            "elo_score_model": elo_score_metrics,
        },
        "baselines": {
            "majority_class": majority_metrics,
            "historical_strength": {
                **strength_metrics,
                "strength_threshold": round(strength_threshold, 4),
            },
        },
        "frontend_parameters": {
            "feature_order": FEATURE_COLUMNS,
            "class_labels": logistic.classes_.tolist(),
            "scaler_mean": [round(float(value), 8) for value in scaler.mean_],
            "scaler_scale": [round(float(value), 8) for value in scaler.scale_],
            "coefficients": [
                [round(float(value), 8) for value in row]
                for row in logistic.coef_
            ],
            "intercepts": [round(float(value), 8) for value in logistic.intercept_],
            "elo_score_model": {
                "feature_order": SCORE_FEATURE_COLUMNS,
                "scaler_mean": [round(float(value), 8) for value in score_scaler.mean_],
                "scaler_scale": [round(float(value), 8) for value in score_scaler.scale_],
                "team_a_goal_coefficients": [round(float(value), 8) for value in goal_model_a.coef_],
                "team_a_goal_intercept": round(float(goal_model_a.intercept_), 8),
                "team_b_goal_coefficients": [round(float(value), 8) for value in goal_model_b.coef_],
                "team_b_goal_intercept": round(float(goal_model_b.intercept_), 8),
                "max_goals": 8,
                "draw_margin": round(draw_margin, 4),
            },
        },
    }

    logistic_importance = np.mean(np.abs(logistic.coef_), axis=0)
    feature_importance = {
        "logistic_regression": sorted(
            [
                {"feature": feature, "importance": round(float(value), 6)}
                for feature, value in zip(FEATURE_COLUMNS, logistic_importance)
            ],
            key=lambda item: item["importance"],
            reverse=True,
        ),
        "random_forest": sorted(
            [
                {"feature": feature, "importance": round(float(value), 6)}
                for feature, value in zip(FEATURE_COLUMNS, forest.feature_importances_)
            ],
            key=lambda item: item["importance"],
            reverse=True,
        ),
        "elo_score_model": sorted(
            [
                {
                    "feature": feature,
                    "importance": round(float(abs(value_a) + abs(value_b)) / 2, 6),
                }
                for feature, value_a, value_b in zip(SCORE_FEATURE_COLUMNS, goal_model_a.coef_, goal_model_b.coef_)
            ],
            key=lambda item: item["importance"],
            reverse=True,
        ),
    }

    team_features = {
        team: {
            key: round(float(value), 6)
            for key, value in values.items()
        }
        for team, values in sorted(latest_team_features.items())
    }

    example_rows = pd.concat([test_df.head(8), test_df.tail(4)]).drop_duplicates("match_id")
    prediction_examples = {
        "examples": [prediction_example(row, logistic, scaler) for _, row in example_rows.iterrows()],
        "elo_score_examples": [
            score_prediction_example(row, goal_model_a, goal_model_b, score_scaler)
            for _, row in example_rows.iterrows()
        ],
    }

    outputs = {
        "model_metrics.json": model_metrics,
        "feature_importance.json": feature_importance,
        "team_features.json": team_features,
        "prediction_examples.json": prediction_examples,
    }
    for filename, payload in outputs.items():
        with (OUTPUT_DIR / filename).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")

    print("Training complete")
    print(f"Logistic Regression accuracy: {logistic_metrics['accuracy']}, macro F1: {logistic_metrics['macro_f1']}")
    print(f"Random Forest accuracy: {forest_metrics['accuracy']}, macro F1: {forest_metrics['macro_f1']}")
    print(f"Calibrated Gradient Boosting accuracy: {gradient_metrics['accuracy']}, macro F1: {gradient_metrics['macro_f1']}")
    print(f"Elo + Score accuracy: {elo_score_metrics['accuracy']}, macro F1: {elo_score_metrics['macro_f1']}, Brier: {elo_score_metrics['brier_score']}")
    print(f"Historical Strength accuracy: {strength_metrics['accuracy']}, macro F1: {strength_metrics['macro_f1']}")


if __name__ == "__main__":
    main()
