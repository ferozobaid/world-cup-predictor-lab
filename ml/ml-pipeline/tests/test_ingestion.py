from src.ingestion.clean_matches import clean_matches
from src.ingestion.normalize_teams import normalize_team_name

import pandas as pd


def test_normalize_team_aliases():
    assert normalize_team_name("USA") == "United States"
    assert normalize_team_name("Côte d'Ivoire") == "Ivory Coast"
    assert normalize_team_name("Czechia") == "Czech Republic"


def test_clean_matches_filters_youth_and_old_rows():
    raw = pd.DataFrame(
        [
            ["2017-01-01", "France", "Germany", 1, 0, "Friendly", False],
            ["2020-01-01", "France U20", "Germany", 1, 0, "Friendly", False],
            ["2021-01-01", "USA", "Mexico", 2, 0, "Friendly", False],
        ],
        columns=["date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"],
    )
    cleaned = clean_matches(raw)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["home_team"] == "United States"
    assert cleaned.iloc[0]["target"] == "H"

