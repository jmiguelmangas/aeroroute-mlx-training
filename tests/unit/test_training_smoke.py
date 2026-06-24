from __future__ import annotations

import json
from pathlib import Path

import pytest

from aeroroute_mlx_training.training.smoke import (
    QLoRASmokeConfig,
    build_qlora_command,
    run_qlora_smoke,
)
from aeroroute_mlx_training.training.smoke_data import write_smoke_dataset
from aeroroute_mlx_training.training.smoke_data import write_experiment_dataset
from aeroroute_mlx_training.training.ranked_data import write_ranked_dataset


def _ready_manifest(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    revision = "4f665a4c50ecfe4ecdc34056ab52fe3e3c4abf9e"
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(
        json.dumps(
            {
                "accepted": True,
                "base_model": "mlx-community/gemma-3-text-4b-it-4bit",
                "base_revision": revision,
            }
        )
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "base_model": "mlx-community/gemma-3-text-4b-it-4bit",
                "architecture": "gemma3_text",
                "modality": "text",
                "quantization": "mlx-4bit",
                "base_revision": revision,
                "local_checkpoint": str(checkpoint),
                "license_acceptance_record": str(acceptance),
            }
        )
    )
    return manifest


def test_smoke_dataset_has_all_required_splits(tmp_path: Path) -> None:
    write_smoke_dataset(tmp_path)
    for name in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        messages = json.loads((tmp_path / name).read_text())["messages"]
        assert messages[-1]["role"] == "assistant"


def test_experiment_dataset_is_group_isolated_and_manifested(
    tmp_path: Path,
) -> None:
    report = write_experiment_dataset(tmp_path, total_records=20)
    assert report.split_counts == {"train": 14, "validation": 3, "test": 3}
    groups_by_split = []
    for filename in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        groups_by_split.append(
            {
                item["messages"][1]["content"]
                for item in map(
                    json.loads, (tmp_path / filename).read_text().splitlines()
                )
            }
        )
    assert groups_by_split[0].isdisjoint(groups_by_split[1])
    assert groups_by_split[0].isdisjoint(groups_by_split[2])
    assert groups_by_split[1].isdisjoint(groups_by_split[2])
    assert (tmp_path / "dataset-manifest.json").is_file()
    target = json.loads((tmp_path / "test.jsonl").read_text().splitlines()[0])[
        "messages"
    ][-1]["content"]
    assert "L0" not in target


def test_ranked_dataset_has_winners_and_review_queue(tmp_path: Path) -> None:
    report = write_ranked_dataset(tmp_path, total_records=20)
    assert report.review_status == "needs_human_review"
    example = json.loads((tmp_path / "test.jsonl").read_text().splitlines()[0])
    assert '"selected_candidate"' not in example["messages"][1]["content"]
    assert example["messages"][2]["content"].startswith(
        "Selected candidate: minimum_fuel. Fuel:"
    )
    assert (tmp_path / "review-queue.jsonl").is_file()


def test_build_qlora_command_uses_bounded_quantized_smoke(
    tmp_path: Path,
) -> None:
    manifest = _ready_manifest(tmp_path)
    command = build_qlora_command(
        QLoRASmokeConfig(manifest, tmp_path, tmp_path / "adapter", iterations=2)
    )
    assert "--train" in command
    assert "--test" in command
    assert "--mask-prompt" in command
    assert "--grad-checkpoint" in command
    assert command[command.index("--iters") + 1] == "2"


def test_smoke_refuses_missing_dataset_before_subprocess(
    tmp_path: Path,
) -> None:
    manifest = _ready_manifest(tmp_path)
    with pytest.raises(ValueError, match="dataset is missing"):
        run_qlora_smoke(
            QLoRASmokeConfig(manifest, tmp_path, tmp_path / "adapter")
        )
