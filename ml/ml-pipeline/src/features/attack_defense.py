"""Attack and defense feature derivation."""

from __future__ import annotations


def attack_defense_features(row_features: dict[str, float]) -> dict[str, float]:
    home_scoring = row_features.get("home_rolling_goals_scored", 0.0)
    away_scoring = row_features.get("away_rolling_goals_scored", 0.0)
    home_conceding = row_features.get("home_rolling_goals_conceded", 0.0)
    away_conceding = row_features.get("away_rolling_goals_conceded", 0.0)

    home_attack = 0.65 * home_scoring + 0.35 * row_features.get("home_last_5_match_win_rate", 0.0)
    away_attack = 0.65 * away_scoring + 0.35 * row_features.get("away_last_5_match_win_rate", 0.0)
    home_defense = 1.0 / (1.0 + home_conceding)
    away_defense = 1.0 / (1.0 + away_conceding)

    return {
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
        "expected_goals_proxy": max(0.1, (home_scoring + away_conceding) / 2.0)
        - max(0.1, (away_scoring + home_conceding) / 2.0),
        "expected_goals_against_proxy": max(0.1, (away_scoring + home_conceding) / 2.0),
    }

