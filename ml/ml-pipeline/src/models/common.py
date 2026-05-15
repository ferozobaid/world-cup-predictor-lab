"""Shared modeling helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import CLASS_LABELS
from src.features.feature_builder import get_feature_columns


def split_feature_frame(frame: pd.DataFrame):
    feature_columns = get_feature_columns(frame)
    train = frame[frame["split"] == "train"]
    validation = frame[frame["split"] == "validation"]
    test = frame[frame["split"] == "test"]
    if train.empty:
        raise ValueError("No training rows found for configured chronological split.")
    return feature_columns, train, validation, test


def target_to_index(values) -> np.ndarray:
    mapping = {label: idx for idx, label in enumerate(CLASS_LABELS)}
    return np.array([mapping[value] for value in values])


def align_probabilities(model, probabilities) -> np.ndarray:
    probabilities = np.asarray(probabilities)
    if probabilities.shape[1] == len(CLASS_LABELS):
        return probabilities
    aligned = np.zeros((probabilities.shape[0], len(CLASS_LABELS)))
    for model_idx, class_label in enumerate(model.classes_):
        aligned[:, int(class_label)] = probabilities[:, model_idx]
    return aligned

