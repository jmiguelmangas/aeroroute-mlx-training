"""Deterministic held-out numeric-fidelity evaluation for local adapters."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from aeroroute_mlx_training.evaluation.metrics import (
    _normalize,
    _numeric_tokens,
)


@dataclass(frozen=True, slots=True)
class AdapterPrediction:
    case_id: str
    allowed_numeric_values: tuple[str, ...]
    prediction: str


@dataclass(frozen=True, slots=True)
class AdapterEvaluationReport:
    total_cases: int
    numeric_fidelity: float
    unsupported_claim_rate: float
    promotion_eligible: bool
    promotion_blocker: str
    predictions: tuple[AdapterPrediction, ...]


def evaluate_predictions(
    predictions: tuple[AdapterPrediction, ...],
) -> AdapterEvaluationReport:
    """Score numeric claims; winner selection is intentionally not in this corpus."""
    if not predictions:
        raise ValueError("adapter evaluation requires at least one prediction")
    passes = 0
    unsupported = 0
    for prediction in predictions:
        allowed = {
            _normalize(value) for value in prediction.allowed_numeric_values
        }
        tokens = {
            _normalize(value)
            for value in _numeric_tokens(prediction.prediction)
        }
        passes += tokens == allowed
        unsupported += bool(tokens - allowed)
    total = len(predictions)
    return AdapterEvaluationReport(
        total_cases=total,
        numeric_fidelity=passes / total,
        unsupported_claim_rate=unsupported / total,
        promotion_eligible=False,
        promotion_blocker="winner_correctness_not_evaluated",
        predictions=predictions,
    )


def evaluate_local_adapter(
    checkpoint_path: Path,
    adapter_path: Path,
    data_path: Path,
    max_tokens: int = 80,
) -> AdapterEvaluationReport:
    """Generate deterministic held-out completions using the local MLX adapter."""
    from mlx_lm import generate, load

    model, tokenizer = load(
        str(checkpoint_path), adapter_path=str(adapter_path)
    )
    predictions: list[AdapterPrediction] = []
    for index, line in enumerate(
        (data_path / "test.jsonl").read_text().splitlines()
    ):
        item = json.loads(line)
        messages = item["messages"]
        facts = json.loads(messages[1]["content"])
        prompt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        response = _generate_completion(
            generate, model, tokenizer, prompt, max_tokens
        )
        predictions.append(
            AdapterPrediction(
                case_id=f"held-out-{index + 1}",
                allowed_numeric_values=tuple(
                    str(value) for value in facts.values()
                ),
                prediction=response,
            )
        )
    return evaluate_predictions(tuple(predictions))


def evaluate_template_baseline(data_path: Path) -> AdapterEvaluationReport:
    """Score the deterministic prompt-only template on the same held-out data."""
    predictions: list[AdapterPrediction] = []
    for index, line in enumerate(
        (data_path / "test.jsonl").read_text().splitlines()
    ):
        item = json.loads(line)
        messages = item["messages"]
        facts = json.loads(messages[1]["content"])
        predictions.append(
            AdapterPrediction(
                case_id=f"held-out-{index + 1}",
                allowed_numeric_values=tuple(
                    str(value) for value in facts.values()
                ),
                prediction=messages[-1]["content"],
            )
        )
    return evaluate_predictions(tuple(predictions))


def write_adapter_evaluation(
    path: Path, report: AdapterEvaluationReport
) -> None:
    path.write_text(json.dumps(asdict(report), indent=2) + "\n")


def _generate_completion(
    generate: Callable[..., str],
    model: object,
    tokenizer: object,
    prompt: str,
    max_tokens: int,
) -> str:
    """Use MLX-LM's greedy default sampler for deterministic local scoring."""
    return generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
