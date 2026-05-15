"""Runtime configuration and filesystem paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DateSplits:
    train_start: str = "2018-01-01"
    train_end: str = "2023-12-31"
    validation_start: str = "2024-01-01"
    validation_end: str = "2024-12-31"
    test_start: str = "2025-01-01"


@dataclass(frozen=True)
class EloConfig:
    initial_rating: float = 1500.0
    initial_offense: float = 1500.0
    initial_defense: float = 1500.0
    home_advantage: float = 60.0
    neutral_home_advantage: float = 0.0
    recency_window: int = 5
    confederation_balance: float = 0.98


@dataclass(frozen=True)
class PipelineConfig:
    root_dir: Path = Path(__file__).resolve().parents[1]
    random_state: int = 42
    min_date: str = "2018-01-01"
    date_splits: DateSplits = DateSplits()
    elo: EloConfig = EloConfig()
    worldcup_fixture_url: str = (
        "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
    )

    @property
    def data_dir(self) -> Path:
        return self.root_dir / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def models_dir(self) -> Path:
        return self.root_dir / "models"

    @property
    def outputs_dir(self) -> Path:
        return self.root_dir / "outputs"

    @property
    def metrics_dir(self) -> Path:
        return self.outputs_dir / "metrics"

    @property
    def predictions_dir(self) -> Path:
        return self.outputs_dir / "predictions"

    @property
    def charts_dir(self) -> Path:
        return self.outputs_dir / "charts"

    def ensure_directories(self) -> None:
        for path in [
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.models_dir,
            self.metrics_dir,
            self.predictions_dir,
            self.charts_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


CONFIG = PipelineConfig()

