"""Model artifact persistence with a joblib fallback."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def dump_artifact(artifact: dict[str, Any], path: Path) -> None:
    try:
        import joblib

        joblib.dump(artifact, path)
    except Exception:
        with path.open("wb") as handle:
            pickle.dump(artifact, handle)


def load_artifact(path: str | Path) -> dict[str, Any]:
    try:
        import joblib

        return joblib.load(path)
    except Exception:
        with Path(path).open("rb") as handle:
            return pickle.load(handle)

