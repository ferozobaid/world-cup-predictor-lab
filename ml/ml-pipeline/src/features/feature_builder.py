"""Build chronological feature matrices."""

from __future__ import annotations

import json

import pandas as pd

from src.config import CONFIG, PipelineConfig
from src.features.attack_defense import attack_defense_features
from src.features.elo_features import EloSystem
from src.features.rolling_form import RollingFormState
from src.features.tournament_features import tournament_features
from src.ingestion.load_enrichment import EnrichmentLookup, load_optional_enrichments
from src.ingestion.load_matches import ingest_matches
from src.utils.date_utils import assign_split
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

NON_FEATURE_COLUMNS = {
    "match_id",
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
    "target",
    "split",
}


def build_features(matches: pd.DataFrame, config: PipelineConfig = CONFIG) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"], errors="raise")
    matches = matches.sort_values(["date", "match_id"]).reset_index(drop=True)

    elo = EloSystem(config.elo)
    form = RollingFormState()
    enrichments = EnrichmentLookup(load_optional_enrichments(config))
    rows: list[dict[str, object]] = []

    for _, match in matches.iterrows():
        match_date = pd.Timestamp(match["date"])
        feature_row = match.to_dict()
        feature_row.update(elo.pre_match_features(str(match["home_team"]), str(match["away_team"]), bool(match["neutral"])))
        feature_row.update(enrichments.features(str(match["home_team"]), str(match["away_team"]), match_date))
        feature_row.update(form.features(str(match["home_team"]), "home", match_date))
        feature_row.update(form.features(str(match["away_team"]), "away", match_date))
        feature_row.update(attack_defense_features(feature_row))
        feature_row.update(
            tournament_features(
                match,
                float(feature_row["home_rest_days"]),
                float(feature_row["away_rest_days"]),
            )
        )
        feature_row["split"] = assign_split(match_date, config.date_splits)
        rows.append(feature_row)

        # Updates happen after row creation, preserving pre-match chronology.
        elo.update(match)
        form.update(match)

    feature_frame = pd.DataFrame(rows)
    numeric_columns = get_feature_columns(feature_frame)
    feature_frame[numeric_columns] = feature_frame[numeric_columns].fillna(0.0)
    return feature_frame, elo.team_strengths(), elo.elo_history()


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]


def build_and_save_features(config: PipelineConfig = CONFIG, sample: bool = False) -> pd.DataFrame:
    config.ensure_directories()
    interim_path = config.interim_dir / "matches_clean.csv"
    if sample or not interim_path.exists():
        matches = ingest_matches(config=config, sample=sample)
    else:
        matches = pd.read_csv(interim_path)
    features, strengths, elo_history = build_features(matches, config=config)
    features.to_csv(config.processed_dir / "match_features.csv", index=False)
    strengths.to_csv(config.processed_dir / "team_strengths.csv", index=False)
    elo_history.to_csv(config.processed_dir / "elo_history.csv", index=False)
    metadata = {
        "rows": int(len(features)),
        "feature_columns": get_feature_columns(features),
        "leakage_policy": "All team-state features are emitted before Elo/form updates for the current match.",
    }
    (config.processed_dir / "feature_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Wrote feature matrix with %s rows and %s features", len(features), len(metadata["feature_columns"]))
    return features
