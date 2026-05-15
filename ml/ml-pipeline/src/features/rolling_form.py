"""Rolling form features computed only from previous matches."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class TeamMatchRecord:
    date: pd.Timestamp
    goals_for: int
    goals_against: int
    points: int


@dataclass
class RollingFormState:
    matches: dict[str, list[TeamMatchRecord]] = field(default_factory=lambda: defaultdict(list))

    def features(self, team: str, prefix: str, match_date: pd.Timestamp) -> dict[str, float]:
        records = self.matches.get(team, [])
        output = {
            f"{prefix}_last_3_match_win_rate": self._win_rate(records, 3),
            f"{prefix}_last_5_match_win_rate": self._win_rate(records, 5),
            f"{prefix}_last_10_match_win_rate": self._win_rate(records, 10),
            f"{prefix}_rolling_goals_scored": self._mean(records, "goals_for", 5),
            f"{prefix}_rolling_goals_conceded": self._mean(records, "goals_against", 5),
            f"{prefix}_rolling_goal_difference": self._goal_difference(records, 5),
            f"{prefix}_rolling_clean_sheet_rate": self._clean_sheet_rate(records, 5),
            f"{prefix}_rolling_expected_points": self._mean(records, "points", 5),
            f"{prefix}_weighted_recent_form": self._weighted_form(records),
            f"{prefix}_rest_days": self._rest_days(records, match_date),
        }
        return output

    def update(self, row: pd.Series) -> None:
        date = pd.Timestamp(row["date"])
        home_score = int(row["home_score"])
        away_score = int(row["away_score"])
        home_points = 3 if home_score > away_score else 1 if home_score == away_score else 0
        away_points = 3 if away_score > home_score else 1 if home_score == away_score else 0
        self.matches[str(row["home_team"])].append(TeamMatchRecord(date, home_score, away_score, home_points))
        self.matches[str(row["away_team"])].append(TeamMatchRecord(date, away_score, home_score, away_points))

    @staticmethod
    def _window(records: list[TeamMatchRecord], n: int) -> list[TeamMatchRecord]:
        return records[-n:]

    def _win_rate(self, records: list[TeamMatchRecord], n: int) -> float:
        window = self._window(records, n)
        if not window:
            return 0.0
        return float(sum(record.points == 3 for record in window) / len(window))

    def _mean(self, records: list[TeamMatchRecord], attr: str, n: int) -> float:
        window = self._window(records, n)
        if not window:
            return 0.0
        return float(np.mean([getattr(record, attr) for record in window]))

    def _goal_difference(self, records: list[TeamMatchRecord], n: int) -> float:
        window = self._window(records, n)
        if not window:
            return 0.0
        return float(np.mean([record.goals_for - record.goals_against for record in window]))

    def _clean_sheet_rate(self, records: list[TeamMatchRecord], n: int) -> float:
        window = self._window(records, n)
        if not window:
            return 0.0
        return float(sum(record.goals_against == 0 for record in window) / len(window))

    def _weighted_form(self, records: list[TeamMatchRecord]) -> float:
        window = self._window(records, 10)
        if not window:
            return 0.0
        weights = np.exp(np.linspace(-1.2, 0.0, len(window)))
        points = np.array([record.points / 3.0 for record in window])
        return float(np.average(points, weights=weights))

    def _rest_days(self, records: list[TeamMatchRecord], match_date: pd.Timestamp) -> float:
        if not records:
            return 30.0
        days = (match_date - records[-1].date).days
        return float(max(0, min(days, 60)))

