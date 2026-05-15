"""Team-name normalization."""

from __future__ import annotations

import re
import unicodedata

from src.constants import TEAM_ALIASES


def normalize_team_name(value: object) -> str:
    if value is None:
        return ""
    name = str(value).strip()
    name = re.sub(r"\s+", " ", name)
    name = name.replace("’", "'").replace("`", "'")
    if name in TEAM_ALIASES:
        return TEAM_ALIASES[name]
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return TEAM_ALIASES.get(ascii_name, name)


def normalize_team_columns(df):
    output = df.copy()
    for column in ["home_team", "away_team"]:
        if column in output.columns:
            output[column] = output[column].map(normalize_team_name)
    return output

