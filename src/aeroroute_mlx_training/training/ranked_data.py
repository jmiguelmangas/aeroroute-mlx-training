"""Synthetic, review-ready route-comparison records with known winners."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from aeroroute_mlx_training.training.smoke_data import _synthetic_icao


@dataclass(frozen=True, slots=True)
class RankedDatasetReport:
    seed: int
    total_records: int
    split_counts: dict[str, int]
    split_sha256: dict[str, str]
    review_status: str


def write_ranked_dataset(
    output_path: Path, total_records: int = 100, seed: int = 42
) -> RankedDatasetReport:
    """Write group-isolated chat data with candidates and a deterministic winner."""
    if total_records < 20:
        raise ValueError("ranked dataset requires at least 20 records")
    output_path.mkdir(parents=True, exist_ok=True)
    targets = {
        "train": total_records * 70 // 100,
        "valid": total_records * 15 // 100,
    }
    targets["test"] = total_records - sum(targets.values())
    splits: dict[str, list[str]] = {name: [] for name in targets}
    index = seed
    while any(len(splits[name]) < targets[name] for name in targets):
        split = ("train", "valid", "test")[
            int(hashlib.sha256(f"{index}".encode()).hexdigest()[:8], 16)
            % 100
            // 80
        ]
        if split == "valid":
            split = "valid"
        if (
            split == "test"
            or int(hashlib.sha256(f"{index}".encode()).hexdigest()[:8], 16)
            % 100
            >= 90
        ):
            split = "test"
        if len(splits[split]) < targets[split]:
            splits[split].append(json.dumps(_scenario(index)))
        index += 1
    contents = {name: "\n".join(rows) + "\n" for name, rows in splits.items()}
    for name, content in contents.items():
        (output_path / f"{name}.jsonl").write_text(content)
    report = RankedDatasetReport(
        seed,
        total_records,
        {name: len(rows) for name, rows in splits.items()},
        {
            name: hashlib.sha256(content.encode()).hexdigest()
            for name, content in contents.items()
        },
        "needs_human_review",
    )
    (output_path / "dataset-manifest.json").write_text(
        json.dumps(asdict(report), indent=2) + "\n"
    )
    (output_path / "review-queue.jsonl").write_text(contents["test"])
    return report


def _scenario(index: int) -> dict[str, object]:
    fuel = 16000 + (index * 137) % 7000
    minutes = 350 + (index * 11) % 170
    alternatives = [
        {"id": "minimum_fuel", "fuel_kg": fuel, "time_min": minutes},
        {
            "id": "shortest_time",
            "fuel_kg": fuel + 650,
            "time_min": minutes - 18,
        },
        {"id": "balanced", "fuel_kg": fuel + 180, "time_min": minutes - 6},
    ]
    facts = {
        "origin": _synthetic_icao("L", index),
        "destination": _synthetic_icao("K", index * 7),
        "alternatives": alternatives,
        "selection_basis": "lowest estimated fuel",
    }
    target = (
        f"Selected candidate: minimum_fuel. Fuel: {fuel} kg. "
        f"Time: {minutes} minutes."
    )
    return {
        "messages": [
            {
                "role": "system",
                "content": "Choose the alternative with the lowest estimated fuel. State its route id, fuel and time only. Do not number or describe other alternatives. Do not give operational flight advice.",
            },
            {"role": "user", "content": json.dumps(facts, sort_keys=True)},
            {"role": "assistant", "content": target},
        ]
    }
