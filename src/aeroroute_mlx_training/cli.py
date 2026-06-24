"""Reproducible offline evaluation commands."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from aeroroute_mlx_training.evaluation.adapter import (
    evaluate_local_adapter,
    evaluate_template_baseline,
    write_adapter_evaluation,
)
from aeroroute_mlx_training.evaluation.metrics import EvaluationCase, evaluate
from aeroroute_mlx_training.compatibility.probe import (
    preflight_smoke,
    probe_model_manifest,
)
from aeroroute_mlx_training.training.smoke import (
    QLoRASmokeConfig,
    run_qlora_smoke,
)
from aeroroute_mlx_training.training.smoke_data import (
    write_experiment_dataset,
    write_smoke_dataset,
)
from aeroroute_mlx_training.training.ranked_data import write_ranked_dataset


def main() -> None:
    parser = argparse.ArgumentParser(prog="aeroroute-mlx-training")
    commands = parser.add_subparsers(dest="command", required=True)
    baseline = commands.add_parser("evaluate-baseline")
    baseline.add_argument("--input", type=Path, required=True)
    baseline.add_argument("--output", type=Path, required=True)
    compatibility = commands.add_parser("probe-compatibility")
    compatibility.add_argument("--manifest", type=Path, required=True)
    preflight = commands.add_parser("preflight-smoke")
    preflight.add_argument("--manifest", type=Path, required=True)
    prepare_data = commands.add_parser("prepare-smoke-data")
    prepare_data.add_argument("--output", type=Path, required=True)
    prepare_experiment = commands.add_parser("prepare-experiment-data")
    prepare_experiment.add_argument("--output", type=Path, required=True)
    prepare_experiment.add_argument("--records", type=int, default=100)
    prepare_experiment.add_argument("--seed", type=int, default=42)
    prepare_ranked = commands.add_parser("prepare-ranked-data")
    prepare_ranked.add_argument("--output", type=Path, required=True)
    prepare_ranked.add_argument("--records", type=int, default=100)
    prepare_ranked.add_argument("--seed", type=int, default=42)
    smoke = commands.add_parser("run-qlora-smoke")
    smoke.add_argument("--manifest", type=Path, required=True)
    smoke.add_argument("--data", type=Path, required=True)
    smoke.add_argument("--adapter-path", type=Path, required=True)
    smoke.add_argument("--steps", type=int, default=50)
    adapter_evaluation = commands.add_parser("evaluate-qlora-adapter")
    adapter_evaluation.add_argument("--checkpoint", type=Path, required=True)
    adapter_evaluation.add_argument("--adapter-path", type=Path, required=True)
    adapter_evaluation.add_argument("--data", type=Path, required=True)
    adapter_evaluation.add_argument("--output", type=Path, required=True)
    template_evaluation = commands.add_parser("evaluate-template-baseline")
    template_evaluation.add_argument("--data", type=Path, required=True)
    template_evaluation.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "evaluate-baseline":
        cases = tuple(
            EvaluationCase(**item)
            for item in json.loads(arguments.input.read_text())
        )
        report = evaluate(cases)
        arguments.output.write_text(json.dumps(asdict(report), indent=2) + "\n")
        print(json.dumps(asdict(report), sort_keys=True))
    if arguments.command == "probe-compatibility":
        report = probe_model_manifest(arguments.manifest)
        print(json.dumps(asdict(report), sort_keys=True))
        if not report.compatible:
            raise SystemExit(2)
    if arguments.command == "preflight-smoke":
        report = preflight_smoke(arguments.manifest)
        print(json.dumps(asdict(report), sort_keys=True))
        if not report.ready:
            raise SystemExit(2)
    if arguments.command == "prepare-smoke-data":
        write_smoke_dataset(arguments.output)
        print(json.dumps({"output": str(arguments.output)}, sort_keys=True))
    if arguments.command == "prepare-experiment-data":
        report = write_experiment_dataset(
            arguments.output, arguments.records, arguments.seed
        )
        print(json.dumps(asdict(report), sort_keys=True))
    if arguments.command == "prepare-ranked-data":
        report = write_ranked_dataset(
            arguments.output, arguments.records, arguments.seed
        )
        print(json.dumps(asdict(report), sort_keys=True))
    if arguments.command == "run-qlora-smoke":
        run_qlora_smoke(
            QLoRASmokeConfig(
                manifest_path=arguments.manifest,
                data_path=arguments.data,
                adapter_path=arguments.adapter_path,
                iterations=arguments.steps,
            )
        )
    if arguments.command == "evaluate-qlora-adapter":
        report = evaluate_local_adapter(
            arguments.checkpoint, arguments.adapter_path, arguments.data
        )
        write_adapter_evaluation(arguments.output, report)
        print(
            json.dumps(
                {
                    "numeric_fidelity": report.numeric_fidelity,
                    "promotion_eligible": report.promotion_eligible,
                    "total_cases": report.total_cases,
                    "unsupported_claim_rate": report.unsupported_claim_rate,
                },
                sort_keys=True,
            )
        )
    if arguments.command == "evaluate-template-baseline":
        report = evaluate_template_baseline(arguments.data)
        write_adapter_evaluation(arguments.output, report)
        print(
            json.dumps(
                {
                    "numeric_fidelity": report.numeric_fidelity,
                    "promotion_eligible": report.promotion_eligible,
                    "total_cases": report.total_cases,
                    "unsupported_claim_rate": report.unsupported_claim_rate,
                },
                sort_keys=True,
            )
        )
