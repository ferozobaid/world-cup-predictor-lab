import json

from src.config import PipelineConfig
from src.simulation.worldcup import simulate_worldcup


def test_simulation_skips_when_fixture_missing(tmp_path):
    config = PipelineConfig(root_dir=tmp_path)
    result = simulate_worldcup(config=config, runs=5)

    assert result["status"] == "skipped"
    output = tmp_path / "outputs" / "predictions" / "worldcup_simulation.json"
    assert json.loads(output.read_text())["data"]["status"] == "skipped"

