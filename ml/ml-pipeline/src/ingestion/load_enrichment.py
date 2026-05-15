"""Optional enrichment data loaders.

Each adapter is local-file first. Missing files produce empty frames so the
baseline pipeline remains usable with only match results.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import CONFIG, PipelineConfig
from src.ingestion.normalize_teams import normalize_team_name


@dataclass
class OptionalEnrichments:
    fifa_rankings: pd.DataFrame
    external_elo: pd.DataFrame
    market_values: pd.DataFrame


def load_optional_enrichments(config: PipelineConfig = CONFIG) -> OptionalEnrichments:
    return OptionalEnrichments(
        fifa_rankings=_load_team_timeseries(config.raw_dir / "fifa_rankings.csv", ["rank", "points"]),
        external_elo=_load_team_timeseries(config.raw_dir / "external_elo.csv", ["elo"]),
        market_values=_load_team_timeseries(config.raw_dir / "market_values.csv", ["market_value_eur"]),
    )


def _load_team_timeseries(path, value_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "team", *value_columns])
    frame = pd.read_csv(path)
    if "date" not in frame.columns or "team" not in frame.columns:
        raise ValueError(f"{path} must include date and team columns")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["team"] = frame["team"].map(normalize_team_name)
    for column in value_columns:
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["date", "team"]).sort_values(["team", "date"]).reset_index(drop=True)


class EnrichmentLookup:
    def __init__(self, enrichments: OptionalEnrichments):
        self.enrichments = enrichments

    def features(self, home_team: str, away_team: str, match_date: pd.Timestamp) -> dict[str, float]:
        home_fifa = self._latest(self.enrichments.fifa_rankings, home_team, match_date)
        away_fifa = self._latest(self.enrichments.fifa_rankings, away_team, match_date)
        home_ext_elo = self._latest(self.enrichments.external_elo, home_team, match_date)
        away_ext_elo = self._latest(self.enrichments.external_elo, away_team, match_date)
        home_value = self._latest(self.enrichments.market_values, home_team, match_date)
        away_value = self._latest(self.enrichments.market_values, away_team, match_date)

        home_rank = float(home_fifa.get("rank", 0.0))
        away_rank = float(away_fifa.get("rank", 0.0))
        home_market = float(home_value.get("market_value_eur", 0.0))
        away_market = float(away_value.get("market_value_eur", 0.0))
        return {
            "home_fifa_rank": home_rank,
            "away_fifa_rank": away_rank,
            "fifa_rank_diff": away_rank - home_rank if home_rank and away_rank else 0.0,
            "home_fifa_points": float(home_fifa.get("points", 0.0)),
            "away_fifa_points": float(away_fifa.get("points", 0.0)),
            "fifa_points_diff": float(home_fifa.get("points", 0.0)) - float(away_fifa.get("points", 0.0)),
            "home_external_elo": float(home_ext_elo.get("elo", 1500.0)),
            "away_external_elo": float(away_ext_elo.get("elo", 1500.0)),
            "external_elo_diff": float(home_ext_elo.get("elo", 1500.0)) - float(away_ext_elo.get("elo", 1500.0)),
            "home_market_value_eur": home_market,
            "away_market_value_eur": away_market,
            "market_value_ratio": home_market / away_market if away_market else 0.0,
        }

    @staticmethod
    def _latest(frame: pd.DataFrame, team: str, match_date: pd.Timestamp) -> dict[str, float]:
        if frame.empty:
            return {}
        records = frame[(frame["team"] == team) & (frame["date"] < match_date)]
        if records.empty:
            return {}
        return records.iloc[-1].to_dict()

