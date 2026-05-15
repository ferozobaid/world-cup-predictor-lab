"""2026 World Cup fixture loading and Monte Carlo simulation."""

from __future__ import annotations

import json
import math
import random
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from src.config import CONFIG, PipelineConfig
from src.constants import CLASS_LABELS
from src.exports.export_frontend_json import _write
from src.ingestion.normalize_teams import normalize_team_name


def fixture_path(config: PipelineConfig = CONFIG) -> Path:
    return config.raw_dir / "worldcup_2026" / "worldcup.json"


def download_worldcup_fixture(config: PipelineConfig = CONFIG) -> Path:
    path = fixture_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(config.worldcup_fixture_url, timeout=30) as response:
        path.write_bytes(response.read())
    return path


def load_worldcup_fixture(config: PipelineConfig = CONFIG) -> dict:
    path = fixture_path(config)
    if not path.exists():
        raise FileNotFoundError(f"World Cup fixture file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_strengths(config: PipelineConfig) -> dict[str, float]:
    path = config.processed_dir / "team_strengths.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    return {str(row["team"]): float(row["elo"]) for _, row in frame.iterrows()}


def _match_probs(team_a: str, team_b: str, strengths: dict[str, float]) -> list[float]:
    elo_a = strengths.get(team_a, 1500.0)
    elo_b = strengths.get(team_b, 1500.0)
    diff = elo_a - elo_b
    home = 1.0 / (1.0 + math.exp(-diff / 260.0))
    draw = max(0.18, 0.28 - abs(diff) / 2500.0)
    decisive = 1.0 - draw
    p_home = decisive * home
    p_away = decisive * (1.0 - home)
    total = p_home + draw + p_away
    return [p_home / total, draw / total, p_away / total]


def _simulate_result(team_a: str, team_b: str, strengths: dict[str, float], rng: random.Random) -> tuple[str, str, int, int]:
    probs = _match_probs(team_a, team_b, strengths)
    result = rng.choices(CLASS_LABELS, weights=probs, k=1)[0]
    if result == "D":
        goals = rng.choice([(0, 0), (1, 1), (2, 2)])
        return "D", "", goals[0], goals[1]
    winner = team_a if result == "H" else team_b
    loser_goals = rng.choice([0, 1, 1, 2])
    winner_goals = loser_goals + rng.choice([1, 1, 2, 3])
    return result, winner, winner_goals if result == "H" else loser_goals, loser_goals if result == "H" else winner_goals


def _group_stage(matches: list[dict], strengths: dict[str, float], rng: random.Random):
    tables = defaultdict(lambda: defaultdict(lambda: {"points": 0, "gf": 0, "ga": 0, "gd": 0}))
    for match in matches:
        group = match.get("group")
        if not group:
            continue
        home = normalize_team_name(match.get("team1"))
        away = normalize_team_name(match.get("team2"))
        if not home or not away or home.startswith(("W", "L")) or away.startswith(("W", "L")):
            continue
        result, _, home_goals, away_goals = _simulate_result(home, away, strengths, rng)
        for team, gf, ga in [(home, home_goals, away_goals), (away, away_goals, home_goals)]:
            tables[group][team]["gf"] += gf
            tables[group][team]["ga"] += ga
            tables[group][team]["gd"] = tables[group][team]["gf"] - tables[group][team]["ga"]
        if result == "H":
            tables[group][home]["points"] += 3
        elif result == "A":
            tables[group][away]["points"] += 3
        else:
            tables[group][home]["points"] += 1
            tables[group][away]["points"] += 1

    qualified = []
    third_place = []
    for group, table in tables.items():
        ranked = sorted(table.items(), key=lambda item: (item[1]["points"], item[1]["gd"], item[1]["gf"]), reverse=True)
        qualified.extend([team for team, _ in ranked[:2]])
        if len(ranked) > 2:
            third_place.append(ranked[2])
    third_ranked = sorted(third_place, key=lambda item: (item[1]["points"], item[1]["gd"], item[1]["gf"]), reverse=True)
    qualified.extend([team for team, _ in third_ranked[:8]])
    return qualified[:32], tables


def _knockout(teams: list[str], strengths: dict[str, float], rng: random.Random, counters: dict[str, Counter]) -> str:
    bracket = teams[:]
    rng.shuffle(bracket)
    round_names = {32: "round_of_32", 16: "quarterfinal", 8: "semifinal", 4: "finalist"}
    while len(bracket) > 1:
        stage_name = round_names.get(len(bracket))
        next_round = []
        for idx in range(0, len(bracket), 2):
            team_a = bracket[idx]
            team_b = bracket[idx + 1]
            result, winner, _, _ = _simulate_result(team_a, team_b, strengths, rng)
            if result == "D":
                winner = rng.choice([team_a, team_b])
            if stage_name:
                counters[stage_name][team_a] += 1
                counters[stage_name][team_b] += 1
            next_round.append(winner)
        bracket = next_round
    counters["champion"][bracket[0]] += 1
    return bracket[0]


def simulate_worldcup(config: PipelineConfig = CONFIG, runs: int = 1000, seed: int | None = None) -> dict:
    config.ensure_directories()
    output_path = config.predictions_dir / "worldcup_simulation.json"
    try:
        fixture = load_worldcup_fixture(config)
    except FileNotFoundError as exc:
        data = {"status": "skipped", "reason": str(exc), "runs": 0}
        _write(output_path, "worldcup_simulation.json", data, {"fixture": str(fixture_path(config))})
        return data

    rng = random.Random(CONFIG.random_state if seed is None else seed)
    strengths = _load_strengths(config)
    matches = fixture.get("matches", [])
    counters = defaultdict(Counter)
    for _ in range(runs):
        qualified, _ = _group_stage(matches, strengths, rng)
        for team in qualified:
            counters["knockout"][team] += 1
        if len(qualified) >= 32:
            _knockout(qualified[:32], strengths, rng, counters)

    teams = sorted(set().union(*[set(counter.keys()) for counter in counters.values()]) if counters else set())
    probabilities = {
        team: {
            "knockout_probability": counters["knockout"][team] / runs,
            "quarterfinal_probability": counters["quarterfinal"][team] / runs,
            "semifinal_probability": counters["semifinal"][team] / runs,
            "finalist_probability": counters["finalist"][team] / runs,
            "champion_probability": counters["champion"][team] / runs,
        }
        for team in teams
    }
    data = {"status": "complete", "runs": runs, "probabilities": probabilities}
    _write(output_path, "worldcup_simulation.json", data, {"fixture": str(fixture_path(config))})
    return data

