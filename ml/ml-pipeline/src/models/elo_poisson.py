"""Elo plus Poisson hybrid baseline."""

from __future__ import annotations

import math

import numpy as np


class EloPoissonHybridModel:
    classes_ = np.array([0, 1, 2])

    def __init__(self, max_goals: int = 8):
        self.max_goals = max_goals
        self.base_home_goals = 1.35
        self.base_away_goals = 1.05

    def fit(self, X, y=None):
        if "home_score" in X and "away_score" in X:
            self.base_home_goals = max(0.5, float(np.mean(X["home_score"])))
            self.base_away_goals = max(0.4, float(np.mean(X["away_score"])))
        return self

    def predict_proba(self, X):
        rows = []
        for _, row in X.iterrows():
            elo_diff = float(row.get("elo_difference", 0.0))
            xg_diff = float(row.get("expected_goals_proxy", 0.0))
            home_lambda = max(0.15, self.base_home_goals * math.exp(elo_diff / 900.0 + xg_diff / 6.0))
            away_lambda = max(0.15, self.base_away_goals * math.exp(-elo_diff / 900.0 - xg_diff / 6.0))
            rows.append(self._match_probabilities(home_lambda, away_lambda))
        return np.asarray(rows)

    def _match_probabilities(self, home_lambda: float, away_lambda: float) -> list[float]:
        home = draw = away = 0.0
        for home_goals in range(self.max_goals + 1):
            p_home = _poisson_pmf(home_goals, home_lambda)
            for away_goals in range(self.max_goals + 1):
                probability = p_home * _poisson_pmf(away_goals, away_lambda)
                if home_goals > away_goals:
                    home += probability
                elif home_goals == away_goals:
                    draw += probability
                else:
                    away += probability
        total = home + draw + away
        return [home / total, draw / total, away / total]


def _poisson_pmf(k: int, rate: float) -> float:
    return math.exp(-rate) * (rate**k) / math.factorial(k)
