"""Strict validation for public/synthetic explanation training records."""

from __future__ import annotations

from aeroroute_mlx_training.dataset_generation.records import ExplanationRecord


def validate_record(record: ExplanationRecord) -> None:
    if record.contract_version != "1.0.0":
        raise ValueError("unsupported explanation contract version")
    if not record.group_id or not record.target:
        raise ValueError("record group and target are required")
    required = ("fuel_kg", "time_min")
    missing = set(required) - set(record.facts)
    if missing:
        raise ValueError("record facts are missing required numeric values")
    for key in required:
        value = record.facts[key]
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"record fact {key} must be non-negative numeric")
        if str(value) not in record.target:
            raise ValueError(
                f"target contains an unsupported or missing {key} claim"
            )
