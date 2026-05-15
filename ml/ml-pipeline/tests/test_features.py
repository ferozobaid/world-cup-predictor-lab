from src.features.feature_builder import build_features, get_feature_columns
from src.ingestion.clean_matches import build_sample_matches


def test_features_are_chronological_for_first_match():
    matches = build_sample_matches()
    features, strengths, history = build_features(matches)
    first = features.iloc[0]

    assert first["home_elo"] == 1500.0
    assert first["away_elo"] == 1500.0
    assert first["home_last_5_match_win_rate"] == 0.0
    assert first["away_last_5_match_win_rate"] == 0.0
    assert not strengths.empty
    assert not history.empty


def test_feature_columns_exclude_targets_and_metadata():
    features, _, _ = build_features(build_sample_matches())
    columns = get_feature_columns(features)

    assert "target" not in columns
    assert "home_score" not in columns
    assert "away_score" not in columns
    assert "elo_difference" in columns
    assert "match_pressure_index" in columns

