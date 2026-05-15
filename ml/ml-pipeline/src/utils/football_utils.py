"""Football-specific helper functions."""

from __future__ import annotations

import math

from src.constants import CONFEDERATION_BY_TEAM, ELO_K_FACTORS, TOURNAMENT_IMPORTANCE


def categorize_tournament(tournament: object) -> str:
    text = str(tournament or "").strip().lower()
    if "world cup" in text and "qualification" not in text and "qualifier" not in text:
        return "world_cup"
    if "friendly" in text:
        return "friendly"
    if "nations league" in text:
        return "nations_league"
    if "qualification" in text or "qualifier" in text:
        return "qualifier"
    if any(
        token in text
        for token in [
            "euro",
            "copa america",
            "copa américa",
            "africa cup",
            "afcon",
            "asian cup",
            "gold cup",
            "ofc nations",
            "concacaf",
        ]
    ):
        return "continental"
    return "other"


def tournament_importance(tournament: object) -> float:
    return TOURNAMENT_IMPORTANCE[categorize_tournament(tournament)]


def tournament_k_factor(tournament: object) -> float:
    return ELO_K_FACTORS[categorize_tournament(tournament)]


def match_result(home_score: int | float, away_score: int | float) -> str:
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def expected_points_for(home_score: int | float, away_score: int | float, is_home: bool) -> float:
    result = match_result(home_score, away_score)
    if result == "D":
        return 1.0
    if (result == "H" and is_home) or (result == "A" and not is_home):
        return 3.0
    return 0.0


def goal_difference_multiplier(goal_difference: int | float, rating_delta: float) -> float:
    gd = abs(float(goal_difference))
    if gd <= 1:
        return 1.0
    # Common Elo-style margin multiplier, dampened when the favorite wins big.
    return math.log(gd + 1.0) * (2.2 / ((abs(rating_delta) * 0.001) + 2.2))


def confederation_multiplier(home_team: str, away_team: str, balance: float) -> float:
    home_confed = CONFEDERATION_BY_TEAM.get(home_team)
    away_confed = CONFEDERATION_BY_TEAM.get(away_team)
    if home_confed and away_confed and home_confed != away_confed:
        return balance
    return 1.0
