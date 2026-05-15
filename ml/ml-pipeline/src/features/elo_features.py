"""Dynamic Elo feature generation with chronological updates."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.config import EloConfig
from src.utils.football_utils import (
    confederation_multiplier,
    goal_difference_multiplier,
    tournament_k_factor,
)


@dataclass
class TeamEloState:
    rating: float = 1500.0
    offensive: float = 1500.0
    defensive: float = 1500.0
    rating_history: list[float] = field(default_factory=list)


class EloSystem:
    def __init__(self, config: EloConfig):
        self.config = config
        self.teams: dict[str, TeamEloState] = {}
        self.history_rows: list[dict[str, object]] = []

    def _state(self, team: str) -> TeamEloState:
        if team not in self.teams:
            self.teams[team] = TeamEloState(
                rating=self.config.initial_rating,
                offensive=self.config.initial_offense,
                defensive=self.config.initial_defense,
            )
        return self.teams[team]

    def pre_match_features(self, home_team: str, away_team: str, neutral: bool) -> dict[str, float]:
        home = self._state(home_team)
        away = self._state(away_team)
        home_advantage = self.config.neutral_home_advantage if neutral else self.config.home_advantage
        return {
            "home_elo": home.rating,
            "away_elo": away.rating,
            "elo_difference": (home.rating + home_advantage) - away.rating,
            "rolling_elo_delta": self._rolling_delta(home) - self._rolling_delta(away),
            "home_offensive_elo": home.offensive,
            "away_offensive_elo": away.offensive,
            "home_defensive_elo": home.defensive,
            "away_defensive_elo": away.defensive,
            "offensive_elo": home.offensive - away.defensive,
            "defensive_elo": home.defensive - away.offensive,
        }

    def update(self, row: pd.Series) -> None:
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        home_score = int(row["home_score"])
        away_score = int(row["away_score"])
        neutral = bool(row["neutral"])

        home = self._state(home_team)
        away = self._state(away_team)
        home_advantage = self.config.neutral_home_advantage if neutral else self.config.home_advantage
        adjusted_delta = (home.rating + home_advantage) - away.rating
        expected_home = 1.0 / (1.0 + 10.0 ** (-adjusted_delta / 400.0))
        actual_home = 1.0 if home_score > away_score else 0.5 if home_score == away_score else 0.0
        gd_multiplier = goal_difference_multiplier(home_score - away_score, adjusted_delta)
        confed_multiplier = confederation_multiplier(home_team, away_team, self.config.confederation_balance)
        k = tournament_k_factor(row.get("tournament")) * gd_multiplier * confed_multiplier
        rating_change = k * (actual_home - expected_home)

        home.rating_history.append(home.rating)
        away.rating_history.append(away.rating)
        home.rating += rating_change
        away.rating -= rating_change

        # Goal-based attack/defense Elo tracks scoring and prevention separately.
        home.offensive += 6.0 * (home_score - 1.25)
        away.defensive -= 6.0 * (home_score - 1.25)
        away.offensive += 6.0 * (away_score - 1.05)
        home.defensive -= 6.0 * (away_score - 1.05)

        self.history_rows.extend(
            [
                {
                    "date": row["date"],
                    "match_id": row["match_id"],
                    "team": home_team,
                    "elo": home.rating,
                    "offensive_elo": home.offensive,
                    "defensive_elo": home.defensive,
                },
                {
                    "date": row["date"],
                    "match_id": row["match_id"],
                    "team": away_team,
                    "elo": away.rating,
                    "offensive_elo": away.offensive,
                    "defensive_elo": away.defensive,
                },
            ]
        )

    def team_strengths(self) -> pd.DataFrame:
        rows = [
            {
                "team": team,
                "elo": state.rating,
                "offensive_elo": state.offensive,
                "defensive_elo": state.defensive,
            }
            for team, state in sorted(self.teams.items())
        ]
        return pd.DataFrame(rows)

    def elo_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.history_rows)

    def _rolling_delta(self, state: TeamEloState) -> float:
        window = self.config.recency_window
        if len(state.rating_history) < 2:
            return 0.0
        recent = state.rating_history[-window:]
        return state.rating - recent[0]

