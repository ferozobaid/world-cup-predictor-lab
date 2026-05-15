"""Date parsing and chronological split helpers."""

from __future__ import annotations

import pandas as pd

from src.config import DateSplits


def parse_date(value: object) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce", utc=False)
    if pd.isna(parsed):
        return pd.NaT
    return pd.Timestamp(parsed).tz_localize(None) if getattr(parsed, "tzinfo", None) else pd.Timestamp(parsed)


def assign_split(date_value: object, splits: DateSplits) -> str:
    date = parse_date(date_value)
    if pd.isna(date):
        return "unknown"
    if pd.Timestamp(splits.train_start) <= date <= pd.Timestamp(splits.train_end):
        return "train"
    if pd.Timestamp(splits.validation_start) <= date <= pd.Timestamp(splits.validation_end):
        return "validation"
    if date >= pd.Timestamp(splits.test_start):
        return "test"
    return "pretrain"

