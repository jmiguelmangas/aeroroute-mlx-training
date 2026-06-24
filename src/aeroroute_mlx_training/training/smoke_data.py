"""Tiny public synthetic chat dataset for a bounded QLoRA smoke."""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from aeroroute_mlx_training.dataset_generation.records import generate_record
from aeroroute_mlx_training.dataset_validation.validation import validate_record
from aeroroute_mlx_training.splitting.grouped import split_by_group


_RUNS = (
    {
        "origin_icao": "LEMD",
        "destination_icao": "KJFK",
        "aircraft_type": "A350",
        "facts": {"fuel_kg": 18000, "time_min": 420},
    },
    {
        "origin_icao": "EGLL",
        "destination_icao": "KJFK",
        "aircraft_type": "A350",
        "facts": {"fuel_kg": 17500, "time_min": 400},
    },
    {
        "origin_icao": "LFPG",
        "destination_icao": "KJFK",
        "aircraft_type": "A350",
        "facts": {"fuel_kg": 17200, "time_min": 395},
    },
)


@dataclass(frozen=True, slots=True)
class ExperimentDatasetReport:
    seed: int
    total_records: int
    split_counts: dict[str, int]
    split_sha256: dict[str, str]


def write_smoke_dataset(output_path: Path) -> None:
    """Write train/valid/test chat JSONL files from public synthetic facts."""
    output_path.mkdir(parents=True, exist_ok=True)
    splits = zip(("train", "valid", "test"), _RUNS, strict=True)
    for split, run in splits:
        record = generate_record(run)
        example = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Explain synthetic route results. Use only supplied facts "
                        "and do not give operational flight advice."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(record.facts, sort_keys=True),
                },
                {"role": "assistant", "content": record.target},
            ]
        }
        (output_path / f"{split}.jsonl").write_text(json.dumps(example) + "\n")


def write_experiment_dataset(
    output_path: Path, total_records: int = 100, seed: int = 42
) -> ExperimentDatasetReport:
    """Write a larger deterministic, group-isolated synthetic chat corpus."""
    if total_records < 20:
        raise ValueError("experiment dataset requires at least 20 records")
    output_path.mkdir(parents=True, exist_ok=True)
    target_counts = {
        "train": total_records * 70 // 100,
        "validation": total_records * 15 // 100,
        "test": total_records
        - (total_records * 70 // 100)
        - (total_records * 15 // 100),
    }
    buckets: dict[str, list[str]] = {name: [] for name in target_counts}
    index = seed
    while any(
        len(buckets[name]) < count for name, count in target_counts.items()
    ):
        run = {
            "origin_icao": _synthetic_icao("L", index),
            "destination_icao": _synthetic_icao("K", index * 7),
            "aircraft_type": ("A350", "A330", "B789")[index % 3],
            "facts": {
                "fuel_kg": 16000 + (index * 137) % 7000,
                "time_min": 350 + (index * 11) % 170,
            },
        }
        record = generate_record(run)
        validate_record(record)
        split = split_by_group((record,))
        split_name = next(
            name
            for name, records in (
                ("train", split.train),
                ("validation", split.validation),
                ("test", split.test),
            )
            if records
        )
        if len(buckets[split_name]) < target_counts[split_name]:
            buckets[split_name].append(json.dumps(_as_chat_example(record)))
        index += 1
    contents: dict[str, str] = {}
    for split, examples in buckets.items():
        content = "\n".join(examples) + "\n"
        contents[split] = content
        filename = "valid" if split == "validation" else split
        (output_path / f"{filename}.jsonl").write_text(content)
    report = ExperimentDatasetReport(
        seed=seed,
        total_records=total_records,
        split_counts={
            name: len(examples) for name, examples in buckets.items()
        },
        split_sha256={
            name: hashlib.sha256(content.encode()).hexdigest()
            for name, content in contents.items()
        },
    )
    (output_path / "dataset-manifest.json").write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"
    )
    return report


def _as_chat_example(record: object) -> dict[str, object]:
    facts = record.facts
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Explain synthetic route results. Use only supplied facts "
                    "and do not give operational flight advice."
                ),
            },
            {"role": "user", "content": json.dumps(facts, sort_keys=True)},
            {"role": "assistant", "content": record.target},
        ]
    }


def _synthetic_icao(prefix: str, index: int) -> str:
    """Return a synthetic four-letter identifier with no numeric-token leakage."""
    letters = []
    value = index
    for _ in range(3):
        letters.append(chr(ord("A") + value % 26))
        value //= 26
    return prefix + "".join(reversed(letters))
