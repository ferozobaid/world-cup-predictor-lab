#!/usr/bin/env python3
"""Export frontend-ready matchup predictions from the offline ML pipeline.

This script is intentionally offline-only. It reads trained model artifacts from
the separate Python pipeline and writes static JSON for the Next.js app.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


APP_TO_PIPELINE_TEAM = {
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "DR Congo": "Congo DR",
    "USA": "United States",
}

STAGES = ["Group stage", "Round of 32", "Round of 16", "Quarter-final", "Semi-final", "Final"]
STAGE_PRESSURE = {
    "Group stage": 0.0,
    "Round of 32": 0.04,
    "Round of 16": 0.08,
    "Quarter-final": 0.12,
    "Semi-final": 0.16,
    "Final": 0.22,
}
MODEL_ALIASES = {
    "calibrated": "catboost",
    "benchmark": "logistic_regression",
    "elo": "elo_poisson",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline-root",
        type=Path,
        default=Path(os.environ["ML_PIPELINE_ROOT"]) if os.environ.get("ML_PIPELINE_ROOT") else None,
        help="Path to the offline Python ML pipeline root. Can also be provided with ML_PIPELINE_ROOT.",
    )
    parser.add_argument(
        "--app-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the Next.js app root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to app ml/model_outputs/matchup_predictions.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pipeline_root is None:
        raise SystemExit("Pass --pipeline-root or set ML_PIPELINE_ROOT to the offline Python ML pipeline root.")
    pipeline_root = args.pipeline_root.resolve()
    app_root = args.app_root.resolve()
    output_path = args.output or (app_root / "ml/model_outputs/matchup_predictions.json")

    sys.path.insert(0, str(pipeline_root))

    from src.constants import CONFEDERATION_BY_TEAM  # pylint: disable=import-error,import-outside-toplevel

    app_data = json.loads((app_root / "src/data/world-cup-data.json").read_text(encoding="utf-8"))
    active_teams = sorted({team for fixture in app_data["fixtures2026"] for team in (fixture["team1"], fixture["team2"])})

    feature_frame = pd.read_csv(pipeline_root / "data/processed/match_features.csv")
    team_strengths = pd.read_csv(pipeline_root / "data/processed/team_strengths.csv")
    manifest = json.loads((pipeline_root / "models/training_manifest.json").read_text(encoding="utf-8"))
    feature_columns = manifest["feature_columns"]
    medians = feature_frame[feature_columns].median(numeric_only=True).to_dict()
    profiles = build_team_profiles(feature_frame, feature_columns)
    strengths = team_strengths.set_index("team").to_dict(orient="index")
    models = {mode: load_model(pipeline_root, source_name) for mode, source_name in MODEL_ALIASES.items()}
    elo_model = models["elo"]["model"]

    predictions: dict[str, Any] = {}
    for team_a in active_teams:
      for team_b in active_teams:
        if team_a == team_b:
          continue
        for stage in STAGES:
          row_ab = build_feature_row(
              team_a,
              team_b,
              stage,
              feature_columns,
              medians,
              profiles,
              strengths,
              CONFEDERATION_BY_TEAM,
          )
          row_ba = build_feature_row(
              team_b,
              team_a,
              stage,
              feature_columns,
              medians,
              profiles,
              strengths,
              CONFEDERATION_BY_TEAM,
          )
          model_payloads = {}
          expected_goals = neutral_expected_goals(row_ab, row_ba, elo_model)
          is_knockout = stage != "Group stage"

          for mode, artifact in models.items():
              probabilities = neutral_probabilities(artifact["model"], row_ab, row_ba, feature_columns)
              rounded = rounded_probabilities(probabilities)
              favorite = favorite_from_probabilities(team_a, team_b, rounded)
              # Conditional Poisson argmax: pick the scoreline within the result class
              # implied by the probabilities so the rendered score is always consistent.
              if favorite == "Toss-up":
                  if is_knockout:
                      favorite_side = "A" if rounded["teamAWin"] >= rounded["teamBWin"] else "B"
                  else:
                      favorite_side = "D"
              elif favorite == team_a:
                  favorite_side = "A"
              else:
                  favorite_side = "B"
              aligned_score = likely_score_from_xg(
                  expected_goals["teamA"],
                  expected_goals["teamB"],
                  favorite_side,
              )
              model_payloads[mode] = {
                  "probabilities": rounded,
                  "likelyScore": aligned_score,
                  "expectedGoals": {
                      "teamA": round(expected_goals["teamA"], 3),
                      "teamB": round(expected_goals["teamB"], 3),
                  },
                  "favorite": favorite,
                  "confidence": confidence_from_probabilities(rounded),
                  "factors": build_factors(mode, team_a, team_b, row_ab, expected_goals),
              }

          predictions[prediction_key(team_a, team_b, stage)] = {
              "teamA": team_a,
              "teamB": team_b,
              "stage": stage,
              "pipelineTeams": {
                  "teamA": pipeline_team(team_a),
                  "teamB": pipeline_team(team_b),
              },
              "models": model_payloads,
          }

    payload = {
        "metadata": {
            "schemaVersion": "1.0",
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "defaultModel": "calibrated",
            "sourceModels": MODEL_ALIASES,
            "stages": STAGES,
            "teamCount": len(active_teams),
            "predictionCount": len(predictions),
            "pipelineRoot": "external-offline-pipeline",
            "notes": "Static frontend artifact generated offline. Next.js must not load Python models at runtime.",
        },
        "data": {
            "teams": active_teams,
            "predictions": predictions,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {output_path} with {len(predictions):,} matchup rows.")


def load_model(pipeline_root: Path, source_name: str) -> dict[str, Any]:
    artifact = joblib.load(pipeline_root / "models" / f"{source_name}.joblib")
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(f"Unexpected model artifact for {source_name}")
    return artifact


def build_team_profiles(frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, dict[str, float]]:
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.sort_values(["date", "match_id"])
    profiles: dict[str, dict[str, float]] = {}
    base_names = {
        column[len("home_") :]
        for column in feature_columns
        if column.startswith("home_") and f"away_{column[len('home_'):]}" in feature_columns
    }

    for _, row in frame.iterrows():
        for team_column, prefix in (("home_team", "home"), ("away_team", "away")):
            team = str(row[team_column])
            profile = profiles.get(team, {})
            for base_name in base_names:
                source = f"{prefix}_{base_name}"
                if source in row and pd.notna(row[source]):
                    profile[base_name] = float(row[source])
            profiles[team] = profile
    return profiles


def build_feature_row(
    team_a: str,
    team_b: str,
    stage: str,
    feature_columns: list[str],
    medians: dict[str, float],
    profiles: dict[str, dict[str, float]],
    strengths: dict[str, dict[str, float]],
    confederation_by_team: dict[str, str],
) -> dict[str, float]:
    pipeline_a = pipeline_team(team_a)
    pipeline_b = pipeline_team(team_b)
    profile_a = profiles.get(pipeline_a, {})
    profile_b = profiles.get(pipeline_b, {})
    strength_a = strengths.get(pipeline_a, {})
    strength_b = strengths.get(pipeline_b, {})
    row = {column: float(medians.get(column, 0.0) or 0.0) for column in feature_columns}

    row["home_elo"] = float(strength_a.get("elo", 1500.0))
    row["away_elo"] = float(strength_b.get("elo", 1500.0))
    row["elo_difference"] = row["home_elo"] - row["away_elo"]
    row["rolling_elo_delta"] = 0.0
    row["home_offensive_elo"] = float(strength_a.get("offensive_elo", 1500.0))
    row["away_offensive_elo"] = float(strength_b.get("offensive_elo", 1500.0))
    row["home_defensive_elo"] = float(strength_a.get("defensive_elo", 1500.0))
    row["away_defensive_elo"] = float(strength_b.get("defensive_elo", 1500.0))
    row["offensive_elo"] = row["home_offensive_elo"] - row["away_defensive_elo"]
    row["defensive_elo"] = row["home_defensive_elo"] - row["away_offensive_elo"]

    copy_profile(row, "home", profile_a)
    copy_profile(row, "away", profile_b)
    apply_attack_defense_features(row)

    row["fifa_rank_diff"] = row.get("away_fifa_rank", 0.0) - row.get("home_fifa_rank", 0.0)
    row["fifa_points_diff"] = row.get("home_fifa_points", 0.0) - row.get("away_fifa_points", 0.0)
    row["external_elo_diff"] = row.get("home_external_elo", 1500.0) - row.get("away_external_elo", 1500.0)
    row["market_value_ratio"] = (
        row.get("home_market_value_eur", 0.0) / row.get("away_market_value_eur", 0.0)
        if row.get("away_market_value_eur", 0.0)
        else 0.0
    )

    home_confed = confederation_by_team.get(pipeline_a, "UNKNOWN")
    away_confed = confederation_by_team.get(pipeline_b, "UNKNOWN")
    derby = float(home_confed != "UNKNOWN" and home_confed == away_confed)
    knockout = float(stage != "Group stage")
    row["tournament_importance"] = 1.2
    row["knockout_flag"] = knockout
    row["derby_flag"] = derby
    row["host_advantage"] = 0.0
    row["continental_matchup"] = float(home_confed != "UNKNOWN" and away_confed != "UNKNOWN" and home_confed != away_confed)
    row["travel_proxy"] = 0.0
    row["rest_days"] = row.get("home_rest_days", 30.0) - row.get("away_rest_days", 30.0)
    row["match_pressure_index"] = 1.45 + STAGE_PRESSURE[stage] + (0.15 if derby else 0.0)

    return {column: float(row.get(column, 0.0) or 0.0) for column in feature_columns}


def copy_profile(row: dict[str, float], prefix: str, profile: dict[str, float]) -> None:
    for name, value in profile.items():
        row[f"{prefix}_{name}"] = float(value)


def apply_attack_defense_features(row: dict[str, float]) -> None:
    home_scoring = row.get("home_rolling_goals_scored", 0.0)
    away_scoring = row.get("away_rolling_goals_scored", 0.0)
    home_conceding = row.get("home_rolling_goals_conceded", 0.0)
    away_conceding = row.get("away_rolling_goals_conceded", 0.0)
    home_attack = 0.65 * home_scoring + 0.35 * row.get("home_last_5_match_win_rate", 0.0)
    away_attack = 0.65 * away_scoring + 0.35 * row.get("away_last_5_match_win_rate", 0.0)
    home_defense = 1.0 / (1.0 + home_conceding)
    away_defense = 1.0 / (1.0 + away_conceding)
    row.update(
        {
            "home_attack_strength": home_attack,
            "away_attack_strength": away_attack,
            "attack_strength": home_attack - away_attack,
            "home_defensive_strength": home_defense,
            "away_defensive_strength": away_defense,
            "defensive_strength": home_defense - away_defense,
            "home_scoring_rate": home_scoring,
            "away_scoring_rate": away_scoring,
            "scoring_rate": home_scoring - away_scoring,
            "home_conceding_rate": home_conceding,
            "away_conceding_rate": away_conceding,
            "conceding_rate": away_conceding - home_conceding,
            "home_expected_goals_proxy": max(0.1, (home_scoring + away_conceding) / 2.0),
            "away_expected_goals_proxy": max(0.1, (away_scoring + home_conceding) / 2.0),
        }
    )
    row["expected_goals_proxy"] = row["home_expected_goals_proxy"] - row["away_expected_goals_proxy"]
    row["expected_goals_against_proxy"] = row["away_expected_goals_proxy"]


def neutral_probabilities(model: Any, row_ab: dict[str, float], row_ba: dict[str, float], feature_columns: list[str]) -> dict[str, float]:
    frame = pd.DataFrame([row_ab, row_ba], columns=feature_columns)
    probabilities = np.asarray(model.predict_proba(frame))
    # Average A-as-home and B-as-home views to avoid arbitrary Team A home bias.
    team_a = (float(probabilities[0][0]) + float(probabilities[1][2])) / 2.0
    draw = (float(probabilities[0][1]) + float(probabilities[1][1])) / 2.0
    team_b = (float(probabilities[0][2]) + float(probabilities[1][0])) / 2.0
    total = team_a + draw + team_b or 1.0
    return {"teamAWin": team_a / total, "draw": draw / total, "teamBWin": team_b / total}


def rounded_probabilities(probabilities: dict[str, float]) -> dict[str, int]:
    raw_values = [
        ("teamAWin", max(0.0, probabilities["teamAWin"] * 100)),
        ("draw", max(0.0, probabilities["draw"] * 100)),
        ("teamBWin", max(0.0, probabilities["teamBWin"] * 100)),
    ]
    floors = {name: math.floor(value) for name, value in raw_values}
    remainder = 100 - sum(floors.values())
    ranked_remainders = sorted(raw_values, key=lambda item: item[1] - math.floor(item[1]), reverse=True)

    for name, _ in ranked_remainders[: max(0, remainder)]:
        floors[name] += 1

    return {
        "teamAWin": int(floors["teamAWin"]),
        "draw": int(floors["draw"]),
        "teamBWin": int(floors["teamBWin"]),
    }


def neutral_expected_goals(row_ab: dict[str, float], row_ba: dict[str, float], elo_model: Any) -> dict[str, float]:
    ab_home, ab_away = expected_goals_for_row(row_ab, elo_model)
    ba_home, ba_away = expected_goals_for_row(row_ba, elo_model)
    return {
        "teamA": (ab_home + ba_away) / 2.0,
        "teamB": (ab_away + ba_home) / 2.0,
    }


def expected_goals_for_row(row: dict[str, float], elo_model: Any) -> tuple[float, float]:
    base = (float(getattr(elo_model, "base_home_goals", 1.35)) + float(getattr(elo_model, "base_away_goals", 1.05))) / 2.0
    elo_diff = float(row.get("elo_difference", 0.0))
    xg_diff = float(row.get("expected_goals_proxy", 0.0))
    home_lambda = max(0.15, min(5.0, base * math.exp(elo_diff / 900.0 + xg_diff / 6.0)))
    away_lambda = max(0.15, min(5.0, base * math.exp(-elo_diff / 900.0 - xg_diff / 6.0)))
    return home_lambda, away_lambda


def likely_score_from_xg(
    team_a_xg: float,
    team_b_xg: float,
    favorite_side: str = "D",
    max_goals: int = 6,
) -> dict[str, int]:
    """Most likely scoreline projected from expected-goals, conditional on result class.

    ``favorite_side`` filters the Poisson grid to the appropriate outcome subset so the
    displayed scoreline is always consistent with the probability ranking:

    * ``"A"``: most likely scoreline where ``team_a`` outscores ``team_b``.
    * ``"B"``: most likely scoreline where ``team_b`` outscores ``team_a``.
    * ``"D"``: most likely draw.

    Without this filter, Poisson argmax over closely-matched teams collapses to 1-1 for
    ~80% of matchups (the most likely *single* outcome is a draw whenever both rates are
    around 1.0-1.3). That's mathematically right but produces a uniform 1-1 / 2-1
    rendering on the UI; the result-class filter restores football-realistic variety.
    """
    best = {"teamA": 0, "teamB": 0}
    best_probability = -1.0
    for goals_a in range(max_goals + 1):
        pa = poisson_pmf(goals_a, team_a_xg)
        for goals_b in range(max_goals + 1):
            if favorite_side == "A" and goals_a <= goals_b:
                continue
            if favorite_side == "B" and goals_b <= goals_a:
                continue
            if favorite_side == "D" and goals_a != goals_b:
                continue
            probability = pa * poisson_pmf(goals_b, team_b_xg)
            if probability > best_probability:
                best_probability = probability
                best = {"teamA": goals_a, "teamB": goals_b}
    return best


def poisson_pmf(k: int, rate: float) -> float:
    return math.exp(-rate) * (rate**k) / math.factorial(k)


def favorite_from_probabilities(team_a: str, team_b: str, probabilities: dict[str, int]) -> str:
    if abs(probabilities["teamAWin"] - probabilities["teamBWin"]) < 5:
        return "Toss-up"
    return team_a if probabilities["teamAWin"] > probabilities["teamBWin"] else team_b


def confidence_from_probabilities(probabilities: dict[str, int]) -> str:
    spread = abs(probabilities["teamAWin"] - probabilities["teamBWin"])
    if spread > 22:
        return "High"
    if spread < 8:
        return "Low"
    return "Medium"


def align_scoreline(
    score: dict[str, int],
    favorite: str,
    stage: str,
    probabilities: dict[str, int],
    team_a: str,
    team_b: str,
) -> dict[str, int]:
    """Ensure the favourite's score >= underdog's score, preserving the goal differential.

    Previous version forced ``winner = loser + 1`` which flattened the Poisson-argmax
    output to a uniform 1-goal margin (the "every scoreline is 2-1" artifact). Now we
    only swap which team holds the higher count; the gap from ``likely_score_from_xg``
    is left intact.
    """
    is_knockout = stage != "Group stage"
    high = max(score["teamA"], score["teamB"], 1 if is_knockout else 0)
    low = max(0, min(score["teamA"], score["teamB"]))

    if favorite == "Toss-up":
        if not is_knockout:
            # Group-stage toss-up: draws are realistic. Cap at 2-2 so we don't show
            # unrealistic 4-4 draws when both xG are high.
            draw_score = min(high, 2)
            return {"teamA": draw_score, "teamB": draw_score}
        # Knockouts can't end as draws — pick the side with the slight probability edge.
        favorite = team_a if probabilities["teamAWin"] >= probabilities["teamBWin"] else team_b

    if favorite == team_a and score["teamA"] <= score["teamB"]:
        return {"teamA": high, "teamB": low}
    if favorite == team_b and score["teamB"] <= score["teamA"]:
        return {"teamA": low, "teamB": high}
    return score


def build_factors(mode: str, team_a: str, team_b: str, row: dict[str, float], expected_goals: dict[str, float]) -> list[dict[str, str]]:
    model_text = {
        "calibrated": "CatBoost model trained offline on 2018-2026 senior international results.",
        "benchmark": "Logistic Regression benchmark trained offline on the same feature set.",
        "elo": "Elo + Poisson baseline converts team strength into expected goals.",
    }[mode]
    elo_gap = row["elo_difference"]
    xg_gap = expected_goals["teamA"] - expected_goals["teamB"]
    form_gap = row.get("home_weighted_recent_form", 0.0) - row.get("away_weighted_recent_form", 0.0)
    return [
        {"label": model_factor_label(mode), "value": model_text, "impact": "neutral"},
        {
            "label": "Elo difference",
            "value": f"{team_a} {elo_gap:+.0f} Elo-point gap vs {team_b}",
            "impact": factor_impact(elo_gap, 35.0),
        },
        {
            "label": "Expected goals",
            "value": f"{team_a} {expected_goals['teamA']:.2f} xG vs {team_b} {expected_goals['teamB']:.2f} xG",
            "impact": factor_impact(xg_gap, 0.18),
        },
        {
            "label": "Recent form",
            "value": f"{form_gap:+.2f} weighted-form gap",
            "impact": factor_impact(form_gap, 0.08),
        },
        {
            "label": "Neutral venue",
            "value": "Team order is symmetrized so the app does not add arbitrary home advantage.",
            "impact": "neutral",
        },
    ]


def model_factor_label(mode: str) -> str:
    if mode == "calibrated":
        return "Calibrated ML"
    if mode == "benchmark":
        return "ML benchmark"
    return "Elo + score model"


def factor_impact(value: float, threshold: float) -> str:
    if value > threshold:
        return "positive"
    if value < -threshold:
        return "negative"
    return "neutral"


def pipeline_team(team: str) -> str:
    return APP_TO_PIPELINE_TEAM.get(team, team)


def prediction_key(team_a: str, team_b: str, stage: str) -> str:
    return f"{team_a}|{team_b}|{stage}"


if __name__ == "__main__":
    main()
