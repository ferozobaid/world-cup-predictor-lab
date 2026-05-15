"""Tournament and context features."""

from __future__ import annotations

from src.constants import CONFEDERATION_BY_TEAM, HOST_NATIONS_2026
from src.utils.football_utils import categorize_tournament, tournament_importance


def tournament_features(row, home_rest_days: float, away_rest_days: float) -> dict[str, float]:
    tournament = row.get("tournament")
    category = categorize_tournament(tournament)
    home_team = str(row.get("home_team"))
    away_team = str(row.get("away_team"))
    country = str(row.get("country") or "")
    neutral = bool(row.get("neutral"))
    home_confed = CONFEDERATION_BY_TEAM.get(home_team, "UNKNOWN")
    away_confed = CONFEDERATION_BY_TEAM.get(away_team, "UNKNOWN")
    derby_flag = float(home_confed != "UNKNOWN" and home_confed == away_confed)
    host_advantage = float((not neutral and country == home_team) or home_team in HOST_NATIONS_2026)
    return {
        "tournament_importance": tournament_importance(tournament),
        "knockout_flag": float(any(token in str(tournament).lower() for token in ["final", "semi", "quarter", "round of"])),
        "derby_flag": derby_flag,
        "host_advantage": host_advantage,
        "continental_matchup": float(home_confed != "UNKNOWN" and away_confed != "UNKNOWN" and home_confed != away_confed),
        "travel_proxy": 0.0 if neutral else float(country not in {"", home_team, away_team}),
        "rest_days": home_rest_days - away_rest_days,
        "match_pressure_index": tournament_importance(tournament)
        + (0.25 if category == "world_cup" else 0.0)
        + (0.15 if derby_flag else 0.0),
    }

