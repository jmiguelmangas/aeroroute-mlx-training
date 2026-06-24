"""Safe wrappers around the MLX-LM QLoRA command-line interface."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from aeroroute_mlx_training.compatibility.probe import preflight_smoke


@dataclass(frozen=True, slots=True)
class QLoRASmokeConfig:
    manifest_path: Path
    data_path: Path
    adapter_path: Path
    iterations: int = 50
    batch_size: int = 1
    num_layers: int = 8
    grad_accumulation_steps: int = 8
    max_seq_length: int = 512
    seed: int = 42


def build_qlora_command(config: QLoRASmokeConfig) -> tuple[str, ...]:
    """Build the bounded MLX-LM invocation for a quantized local checkpoint."""
    manifest = json.loads(config.manifest_path.read_text())
    checkpoint = str(manifest["local_checkpoint"])
    return (
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        checkpoint,
        "--train",
        "--test",
        "--data",
        str(config.data_path),
        "--fine-tune-type",
        "lora",
        "--mask-prompt",
        "--num-layers",
        str(config.num_layers),
        "--batch-size",
        str(config.batch_size),
        "--iters",
        str(config.iterations),
        "--grad-accumulation-steps",
        str(config.grad_accumulation_steps),
        "--adapter-path",
        str(config.adapter_path),
        "--max-seq-length",
        str(config.max_seq_length),
        "--grad-checkpoint",
        "--seed",
        str(config.seed),
    )


def run_qlora_smoke(config: QLoRASmokeConfig) -> None:
    """Run a bounded QLoRA smoke only when local inputs pass validation."""
    report = preflight_smoke(config.manifest_path)
    if not report.ready:
        reasons = ", ".join(report.reasons)
        raise ValueError(f"QLoRA smoke preflight failed: {reasons}")
    required_files = ("train.jsonl", "valid.jsonl", "test.jsonl")
    missing = [
        name
        for name in required_files
        if not (config.data_path / name).is_file()
    ]
    if missing:
        raise ValueError(
            f"QLoRA smoke dataset is missing: {', '.join(missing)}"
        )
    config.adapter_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(build_qlora_command(config), check=True)
