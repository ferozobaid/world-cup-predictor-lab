from src.config import CONFIG
from src.features.elo_features import EloSystem
from src.ingestion.clean_matches import build_sample_matches


def test_elo_updates_after_match():
    matches = build_sample_matches()
    system = EloSystem(CONFIG.elo)
    before = system.pre_match_features("France", "Croatia", neutral=True)
    system.update(matches.iloc[2])
    after = system.pre_match_features("France", "Croatia", neutral=True)

    assert before["home_elo"] == 1500.0
    assert after["home_elo"] > before["home_elo"]
    assert after["away_elo"] < before["away_elo"]

