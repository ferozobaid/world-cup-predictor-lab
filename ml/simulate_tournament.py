"""Offline Monte Carlo simulator for the 2026 FIFA World Cup.

This script is run **offline** to produce
`ml/model_outputs/worldcup_simulation.json`, which the Next.js frontend imports
statically. It must not be invoked at runtime from the Next.js app.

The simulator uses Elo-style team strengths (from
`ml/model_outputs/team_strengths.json`) to model each match as a categorical
draw over [home_win, draw, away_win]. CatBoost / Calibrated ML probabilities
are NOT consumed here — those are still the source of truth for single-match
predictions in the UI. Tournament Monte Carlo intentionally remains a
strength-based comparison view.

Adapted from the sibling `ml-pipeline/src/simulation/worldcup.py`. Made
self-contained (no external package imports beyond Python stdlib) so this repo
can regenerate the artifact without depending on the offline pipeline.

CLI:
    python ml/simulate_tournament.py --runs 100000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIM_RUNS = 100_000  # default offline simulation count
DEFAULT_SEED = 42
PROGRESS_INTERVAL = 5_000  # log a progress line every N runs

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = REPO_ROOT / "ml" / "data" / "worldcup_2026" / "worldcup.json"
STRENGTHS_PATH = REPO_ROOT / "ml" / "model_outputs" / "team_strengths.json"
OUTPUT_PATH = REPO_ROOT / "ml" / "model_outputs" / "worldcup_simulation.json"

CLASS_LABELS = ["H", "D", "A"]

# Subset of aliases used by the 2026 fixture file. Mirrors
# ml-pipeline/src/constants.py::TEAM_ALIASES.
TEAM_ALIASES: Dict[str, str] = {
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

LOGGER = logging.getLogger("simulate_tournament")


# ---------------------------------------------------------------------------
# Helpers (inlined from the sibling pipeline)
# ---------------------------------------------------------------------------


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


def load_fixture(path: Path = FIXTURE_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"World Cup fixture not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_strengths(path: Path = STRENGTHS_PATH) -> Dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Team strengths not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("Unexpected team_strengths.json shape: expected {data: [...]}")
    strengths: Dict[str, float] = {}
    for row in rows:
        team = row.get("team")
        elo = row.get("elo")
        if team is None or elo is None:
            continue
        strengths[str(team)] = float(elo)
    return strengths


# ---------------------------------------------------------------------------
# Simulation core
# ---------------------------------------------------------------------------


def _match_probs(team_a: str, team_b: str, strengths: Dict[str, float]) -> List[float]:
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


def _simulate_result(
    team_a: str,
    team_b: str,
    strengths: Dict[str, float],
    rng: random.Random,
) -> Tuple[str, str, int, int]:
    probs = _match_probs(team_a, team_b, strengths)
    result = rng.choices(CLASS_LABELS, weights=probs, k=1)[0]
    if result == "D":
        goals = rng.choice([(0, 0), (1, 1), (2, 2)])
        return "D", "", goals[0], goals[1]
    loser_goals = rng.choice([0, 1, 1, 2])
    winner_goals = loser_goals + rng.choice([1, 1, 2, 3])
    if result == "H":
        return "H", team_a, winner_goals, loser_goals
    return "A", team_b, loser_goals, winner_goals


def _group_stage(
    matches: List[dict],
    strengths: Dict[str, float],
    rng: random.Random,
):
    tables: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"points": 0, "gf": 0, "ga": 0, "gd": 0})
    )
    for match in matches:
        group = match.get("group")
        if not group:
            continue
        home = normalize_team_name(match.get("team1"))
        away = normalize_team_name(match.get("team2"))
        if not home or not away:
            continue
        if home.startswith(("W", "L")) or away.startswith(("W", "L")):
            # Placeholder slot (e.g., "Winner Group A") — skip; group stage feeds these.
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

    qualified: List[str] = []
    third_place: List[Tuple[str, Dict[str, int]]] = []
    for _, table in tables.items():
        ranked = sorted(
            table.items(),
            key=lambda item: (item[1]["points"], item[1]["gd"], item[1]["gf"]),
            reverse=True,
        )
        qualified.extend([team for team, _ in ranked[:2]])
        if len(ranked) > 2:
            third_place.append(ranked[2])
    third_ranked = sorted(
        third_place,
        key=lambda item: (item[1]["points"], item[1]["gd"], item[1]["gf"]),
        reverse=True,
    )
    qualified.extend([team for team, _ in third_ranked[:8]])
    return qualified[:32], tables


def _knockout(
    teams: List[str],
    strengths: Dict[str, float],
    rng: random.Random,
    counters: Dict[str, Counter],
) -> str:
    bracket = teams[:]
    rng.shuffle(bracket)
    round_names = {32: "round_of_32", 16: "quarterfinal", 8: "semifinal", 4: "finalist"}
    while len(bracket) > 1:
        stage_name = round_names.get(len(bracket))
        next_round: List[str] = []
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def simulate_worldcup(runs: int = SIM_RUNS, seed: int = DEFAULT_SEED) -> dict:
    """Run `runs` tournament simulations and return the result payload.

    Side-effect: writes the payload to OUTPUT_PATH.
    """
    fixture = load_fixture()
    strengths = load_strengths()
    matches = fixture.get("matches", [])

    LOGGER.info(
        "starting simulation: runs=%d seed=%d teams_with_strength=%d fixture_matches=%d",
        runs,
        seed,
        len(strengths),
        len(matches),
    )

    rng = random.Random(seed)
    counters: Dict[str, Counter] = defaultdict(Counter)
    start = time.perf_counter()
    next_log = PROGRESS_INTERVAL

    for run_index in range(1, runs + 1):
        qualified, _ = _group_stage(matches, strengths, rng)
        for team in qualified:
            counters["knockout"][team] += 1
        if len(qualified) >= 32:
            _knockout(qualified[:32], strengths, rng, counters)
        if run_index == next_log or run_index == runs:
            elapsed = time.perf_counter() - start
            rate = run_index / elapsed if elapsed > 0 else 0.0
            eta = (runs - run_index) / rate if rate > 0 else 0.0
            LOGGER.info(
                "progress: %d/%d runs  elapsed=%.1fs  rate=%.0f runs/s  eta=%.1fs",
                run_index,
                runs,
                elapsed,
                rate,
                eta,
            )
            next_log += PROGRESS_INTERVAL

    wall_clock = time.perf_counter() - start

    teams = sorted(
        set().union(*[set(counter.keys()) for counter in counters.values()])
        if counters
        else set()
    )
    probabilities = {
        team: {
            "knockout_probability": round(counters["knockout"][team] / runs, 6),
            "quarterfinal_probability": round(counters["quarterfinal"][team] / runs, 6),
            "semifinal_probability": round(counters["semifinal"][team] / runs, 6),
            "finalist_probability": round(counters["finalist"][team] / runs, 6),
            "champion_probability": round(counters["champion"][team] / runs, 6),
        }
        for team in teams
    }

    metadata = {
        "schema_version": "1.0.0",
        "model_version": "ml-pipeline-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact": "worldcup_simulation.json",
        "runs": runs,
        "seed": seed,
        "wall_clock_seconds": round(wall_clock, 3),
        "runs_per_second": round(runs / wall_clock, 2) if wall_clock > 0 else None,
        "source": {
            "fixture": str(FIXTURE_PATH.relative_to(REPO_ROOT)),
            "strengths": str(STRENGTHS_PATH.relative_to(REPO_ROOT)),
            "probability_model": "elo-team-strength",
            "notes": (
                "Tournament Monte Carlo is Elo/team-strength-based. "
                "Single-match predictions in the UI use the Calibrated ML "
                "(CatBoost) model — see matchup_predictions.json."
            ),
        },
    }
    payload = {"metadata": metadata, "data": {"status": "complete", "runs": runs, "probabilities": probabilities}}

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    LOGGER.info(
        "wrote %s  runs=%d  teams=%d  wall_clock=%.2fs (%.1f runs/s)",
        OUTPUT_PATH.relative_to(REPO_ROOT),
        runs,
        len(teams),
        wall_clock,
        runs / wall_clock if wall_clock > 0 else 0.0,
    )
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline Monte Carlo for the 2026 World Cup.")
    parser.add_argument("--runs", type=int, default=SIM_RUNS, help=f"Number of simulations (default: {SIM_RUNS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"PRNG seed (default: {DEFAULT_SEED})")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logs.")
    return parser


def _main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    simulate_worldcup(runs=args.runs, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
