"""Leakage-resistant deterministic group split."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from aeroroute_mlx_training.dataset_generation.records import ExplanationRecord

SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    train: tuple[ExplanationRecord, ...]
    validation: tuple[ExplanationRecord, ...]
    test: tuple[ExplanationRecord, ...]


def split_by_group(records: tuple[ExplanationRecord, ...]) -> DatasetSplits:
    buckets: dict[SplitName, list[ExplanationRecord]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for record in records:
        bucket = _split_for_group(record.group_id)
        buckets[bucket].append(record)
    return DatasetSplits(
        tuple(buckets["train"]),
        tuple(buckets["validation"]),
        tuple(buckets["test"]),
    )


def _split_for_group(group_id: str) -> SplitName:
    bucket = int(hashlib.sha256(group_id.encode()).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"
