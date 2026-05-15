"""Canonical match cleaning for public international football datasets."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

from src.config import CONFIG
from src.constants import EXCLUDED_TEAM_PATTERNS, OFFICIAL_OR_SENIOR_TOURNAMENT_KEYWORDS
from src.ingestion.normalize_teams import normalize_team_columns
from src.utils.football_utils import match_result

SOURCE_COLUMN_ALIASES = {
    "date": ["date", "match_date", "kickoff_at"],
    "home_team": ["home_team", "home", "team1", "home_team_name"],
    "away_team": ["away_team", "away", "team2", "away_team_name"],
    "home_score": ["home_score", "home_goals", "score1", "goals1"],
    "away_score": ["away_score", "away_goals", "score2", "goals2"],
    "tournament": ["tournament", "competition", "round", "stage"],
    "city": ["city", "venue_city", "ground"],
    "country": ["country", "host_country"],
    "neutral": ["neutral", "neutral_venue", "is_neutral"],
}


def _first_existing(columns: Iterable[str], candidates: list[str]) -> str | None:
    normalized = {column.lower().strip(): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def normalize_match_columns(raw: pd.DataFrame, source_name: str = "unknown") -> pd.DataFrame:
    mapped = {}
    for canonical, candidates in SOURCE_COLUMN_ALIASES.items():
        source_column = _first_existing(raw.columns, candidates)
        if source_column is not None:
            mapped[canonical] = raw[source_column]

    required = {"date", "home_team", "away_team", "home_score", "away_score"}
    missing = sorted(required.difference(mapped))
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")

    frame = pd.DataFrame(mapped)
    frame["source"] = source_name
    if "tournament" not in frame:
        frame["tournament"] = "Unknown"
    if "city" not in frame:
        frame["city"] = ""
    if "country" not in frame:
        frame["country"] = ""
    if "neutral" not in frame:
        frame["neutral"] = False

    return frame[
        [
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
            "city",
            "country",
            "neutral",
            "source",
        ]
    ]


def _is_allowed_tournament(value: object) -> bool:
    text = str(value or "").lower()
    return any(keyword in text for keyword in OFFICIAL_OR_SENIOR_TOURNAMENT_KEYWORDS)


def _is_excluded_team(value: object) -> bool:
    text = str(value or "")
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in EXCLUDED_TEAM_PATTERNS)


def clean_matches(raw: pd.DataFrame, min_date: str = CONFIG.min_date) -> pd.DataFrame:
    frame = normalize_team_columns(raw)
    if "source" not in frame.columns:
        frame["source"] = "manual"
    for optional_column, default in {"city": "", "country": "", "neutral": False, "tournament": "Unknown"}.items():
        if optional_column not in frame.columns:
            frame[optional_column] = default
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["home_score"] = pd.to_numeric(frame["home_score"], errors="coerce")
    frame["away_score"] = pd.to_numeric(frame["away_score"], errors="coerce")
    frame["neutral"] = frame["neutral"].fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])

    frame = frame.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    frame = frame[frame["date"] >= pd.Timestamp(min_date)]
    frame = frame[frame["home_team"].astype(str).str.len() > 0]
    frame = frame[frame["away_team"].astype(str).str.len() > 0]
    frame = frame[frame["home_team"] != frame["away_team"]]
    frame = frame[frame["tournament"].map(_is_allowed_tournament)]
    frame = frame[~frame["home_team"].map(_is_excluded_team)]
    frame = frame[~frame["away_team"].map(_is_excluded_team)]

    frame["home_score"] = frame["home_score"].astype(int)
    frame["away_score"] = frame["away_score"].astype(int)
    frame["target"] = [match_result(h, a) for h, a in zip(frame["home_score"], frame["away_score"])]
    frame = frame.sort_values(["date", "source", "home_team", "away_team"]).reset_index(drop=True)
    frame.insert(0, "match_id", [f"match_{idx + 1:06d}" for idx in range(len(frame))])
    return frame


def build_sample_matches() -> pd.DataFrame:
    rows = [
        ("2018-03-23", "Germany", "Spain", 1, 1, "Friendly", "Dusseldorf", "Germany", False),
        ("2018-06-17", "Germany", "Mexico", 0, 1, "FIFA World Cup", "Moscow", "Russia", True),
        ("2018-07-15", "France", "Croatia", 4, 2, "FIFA World Cup", "Moscow", "Russia", True),
        ("2019-06-07", "United States", "Mexico", 0, 3, "Friendly", "East Rutherford", "United States", False),
        ("2020-10-11", "France", "Portugal", 0, 0, "UEFA Nations League", "Paris", "France", False),
        ("2021-07-10", "Argentina", "Brazil", 1, 0, "Copa America", "Rio de Janeiro", "Brazil", False),
        ("2022-11-23", "Germany", "Japan", 1, 2, "FIFA World Cup", "Doha", "Qatar", True),
        ("2022-12-18", "Argentina", "France", 3, 3, "FIFA World Cup", "Lusail", "Qatar", True),
        ("2023-09-12", "Germany", "France", 2, 1, "Friendly", "Dortmund", "Germany", False),
        ("2024-06-20", "Spain", "Italy", 1, 0, "UEFA Euro", "Gelsenkirchen", "Germany", True),
        ("2024-07-14", "Spain", "England", 2, 1, "UEFA Euro", "Berlin", "Germany", True),
        ("2025-03-20", "Argentina", "Uruguay", 1, 0, "World Cup qualification", "Buenos Aires", "Argentina", False),
        ("2025-06-10", "United States", "Turkey", 2, 1, "Friendly", "Chicago", "United States", False),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
            "city",
            "country",
            "neutral",
        ],
    )
    frame["source"] = "sample"
    return clean_matches(frame)
