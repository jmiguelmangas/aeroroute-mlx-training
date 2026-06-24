import json
from pathlib import Path

from aeroroute_mlx_training.compatibility.probe import (
    preflight_smoke,
    probe_model_manifest,
)


def test_compatibility_requires_immutable_text_only_mlx_manifest(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "architecture": "gemma3_text",
                "modality": "text",
                "quantization": "mlx-4bit",
                "base_revision": "0123456789abcdef",
            }
        )
    )

    assert probe_model_manifest(manifest).compatible


def test_compatibility_rejects_placeholder_revision(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")

    assert (
        "base_revision_not_immutable" in probe_model_manifest(manifest).reasons
    )


def test_smoke_preflight_requires_matching_local_checkpoint_and_license(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    acceptance = tmp_path / "license-acceptance.json"
    revision = "0123456789abcdef"
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

    assert preflight_smoke(manifest).ready


def test_smoke_preflight_never_allows_missing_local_artifacts(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "architecture": "gemma3_text",
                "modality": "text",
                "quantization": "mlx-4bit",
                "base_revision": "0123456789abcdef",
            }
        )
    )

    report = preflight_smoke(manifest)

    assert not report.ready
    assert report.reasons == (
        "local_checkpoint_not_declared",
        "license_acceptance_record_not_declared",
    )
