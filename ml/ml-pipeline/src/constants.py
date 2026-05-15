"""Shared constants for football modeling."""

from __future__ import annotations

CLASS_LABELS = ["H", "D", "A"]
CLASS_NAMES = {"H": "Home Win", "D": "Draw", "A": "Away Win"}

EXPORT_SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "ml-pipeline-v1"

TEAM_ALIASES = {
    "USA": "United States",
    "U.S.A.": "United States",
    "USMNT": "United States",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Cote d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "DR Congo": "Congo DR",
    "Democratic Republic of Congo": "Congo DR",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Curacao": "Curaçao",
}

EXCLUDED_TEAM_PATTERNS = [
    r"\bU-?\d{2}\b",
    r"\bUnder\s?\d{2}\b",
    r"\bOlympic\b",
    r"\bB\b$",
    r"\bXI\b$",
    r"\bAmateur\b",
    r"\bWomen\b",
]

OFFICIAL_OR_SENIOR_TOURNAMENT_KEYWORDS = [
    "friendly",
    "fifa world cup",
    "world cup",
    "qualification",
    "qualifier",
    "uefa euro",
    "uefa nations league",
    "nations league",
    "copa america",
    "copa américa",
    "africa cup",
    "african cup",
    "afcon",
    "afc asian cup",
    "asian cup",
    "concacaf",
    "gold cup",
    "ofc nations cup",
    "confederations cup",
    "fifa series",
]

TOURNAMENT_IMPORTANCE = {
    "friendly": 0.45,
    "nations_league": 0.70,
    "qualifier": 0.85,
    "continental": 1.00,
    "world_cup": 1.20,
    "other": 0.65,
}

ELO_K_FACTORS = {
    "friendly": 16.0,
    "nations_league": 24.0,
    "qualifier": 32.0,
    "continental": 40.0,
    "world_cup": 50.0,
    "other": 22.0,
}

HOST_NATIONS_2026 = {"Canada", "Mexico", "United States"}

CONFEDERATION_BY_TEAM = {
    "Argentina": "CONMEBOL",
    "Brazil": "CONMEBOL",
    "Colombia": "CONMEBOL",
    "Ecuador": "CONMEBOL",
    "Paraguay": "CONMEBOL",
    "Uruguay": "CONMEBOL",
    "Canada": "CONCACAF",
    "Mexico": "CONCACAF",
    "United States": "CONCACAF",
    "Costa Rica": "CONCACAF",
    "Jamaica": "CONCACAF",
    "Haiti": "CONCACAF",
    "England": "UEFA",
    "France": "UEFA",
    "Germany": "UEFA",
    "Spain": "UEFA",
    "Portugal": "UEFA",
    "Netherlands": "UEFA",
    "Italy": "UEFA",
    "Belgium": "UEFA",
    "Croatia": "UEFA",
    "Morocco": "CAF",
    "Senegal": "CAF",
    "Egypt": "CAF",
    "Ghana": "CAF",
    "Ivory Coast": "CAF",
    "Japan": "AFC",
    "South Korea": "AFC",
    "Australia": "AFC",
    "Iran": "AFC",
    "Qatar": "AFC",
    "New Zealand": "OFC",
}
