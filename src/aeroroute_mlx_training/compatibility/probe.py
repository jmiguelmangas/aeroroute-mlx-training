"""Preflight gate for local Gemma/MLX compatibility experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    compatible: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SmokePreflightReport:
    ready: bool
    reasons: tuple[str, ...]


def probe_model_manifest(manifest_path: Path) -> CompatibilityReport:
    manifest = json.loads(manifest_path.read_text())
    reasons: list[str] = []
    if manifest.get("architecture") != "gemma3_text":
        reasons.append("unsupported_architecture")
    if manifest.get("modality") != "text":
        reasons.append("unsupported_modality")
    if manifest.get("quantization") != "mlx-4bit":
        reasons.append("unsupported_quantization")
    revision = manifest.get("base_revision", "")
    if (
        not isinstance(revision, str)
        or len(revision) < 12
        or "REPLACE" in revision
    ):
        reasons.append("base_revision_not_immutable")
    return CompatibilityReport(not reasons, tuple(reasons))


def preflight_smoke(manifest_path: Path) -> SmokePreflightReport:
    """Verify local, licensed inputs before a QLoRA smoke may start.

    The manifest intentionally refers to files outside this repository. Model
    weights and licence acknowledgements are user-supplied, local artefacts and
    must never be generated or downloaded by this command.
    """
    compatibility = probe_model_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    reasons = list(compatibility.reasons)

    checkpoint_path = manifest.get("local_checkpoint")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        reasons.append("local_checkpoint_not_declared")
    elif not Path(checkpoint_path).is_dir():
        reasons.append("local_checkpoint_not_found")
    elif not any(Path(checkpoint_path).glob("*.safetensors")):
        reasons.append("local_checkpoint_weights_not_found")

    acceptance_path = manifest.get("license_acceptance_record")
    if not isinstance(acceptance_path, str) or not acceptance_path:
        reasons.append("license_acceptance_record_not_declared")
    elif not Path(acceptance_path).is_file():
        reasons.append("license_acceptance_record_not_found")
    else:
        try:
            acceptance = json.loads(Path(acceptance_path).read_text())
        except json.JSONDecodeError:
            reasons.append("license_acceptance_record_invalid")
        else:
            if acceptance.get("accepted") is not True:
                reasons.append("license_not_accepted")
            if acceptance.get("base_model") != manifest.get("base_model"):
                reasons.append("license_model_mismatch")
            if acceptance.get("base_revision") != manifest.get("base_revision"):
                reasons.append("license_revision_mismatch")

    return SmokePreflightReport(not reasons, tuple(reasons))
