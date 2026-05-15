import json

from src.config import CONFIG
from src.exports.export_frontend_json import _write


def test_export_schema_validates(tmp_path):
    path = tmp_path / "artifact.json"
    _write(path, "artifact.json", {"ok": True}, {"test": True})
    payload = json.loads(path.read_text())

    assert payload["metadata"]["schema_version"]
    assert payload["metadata"]["artifact"] == "artifact.json"
    assert payload["data"]["ok"] is True


def test_config_points_to_pipeline_root():
    assert CONFIG.root_dir.name == "ml-pipeline"

