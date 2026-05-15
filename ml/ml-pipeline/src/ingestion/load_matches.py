"""Load raw match files and write canonical interim data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import CONFIG, PipelineConfig
from src.ingestion.clean_matches import build_sample_matches, clean_matches, normalize_match_columns
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def find_raw_match_files(config: PipelineConfig = CONFIG) -> list[Path]:
    excluded_parts = {"worldcup_2026"}
    files = []
    for path in sorted(config.raw_dir.rglob("*.csv")):
        if any(part in excluded_parts for part in path.parts):
            continue
        files.append(path)
    return files


def load_raw_matches(config: PipelineConfig = CONFIG) -> pd.DataFrame:
    frames = []
    for path in find_raw_match_files(config):
        raw = pd.read_csv(path)
        frames.append(normalize_match_columns(raw, source_name=path.name))
    if not frames:
        raise FileNotFoundError(
            f"No raw match CSV files found in {config.raw_dir}. "
            "Place the Kaggle results CSV there, or run with --sample."
        )
    return pd.concat(frames, ignore_index=True)


def ingest_matches(config: PipelineConfig = CONFIG, sample: bool = False) -> pd.DataFrame:
    config.ensure_directories()
    matches = build_sample_matches() if sample else clean_matches(load_raw_matches(config))
    output_path = config.interim_dir / "matches_clean.csv"
    matches.to_csv(output_path, index=False)
    logger.info("Wrote %s cleaned matches to %s", len(matches), output_path)
    return matches

