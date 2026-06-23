"""Deterministic evaluation metrics for explanation configurations."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    expected_winner: str
    allowed_numeric_values: tuple[str, ...]
    prediction: str
    predicted_winner: str


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    total_cases: int
    winner_correctness: float
    numeric_fidelity: float
    unsupported_claim_rate: float


def evaluate(cases: tuple[EvaluationCase, ...]) -> EvaluationReport:
    if not cases:
        raise ValueError("evaluation requires at least one case")
    winner_matches = sum(
        case.expected_winner == case.predicted_winner for case in cases
    )
    numeric_passes = 0
    unsupported_claims = 0
    for case in cases:
        allowed = {_normalize(value) for value in case.allowed_numeric_values}
        tokens = {
            _normalize(token) for token in _numeric_tokens(case.prediction)
        }
        if tokens <= allowed:
            numeric_passes += 1
        if tokens - allowed:
            unsupported_claims += 1
    total = len(cases)
    return EvaluationReport(
        total,
        winner_matches / total,
        numeric_passes / total,
        unsupported_claims / total,
    )


def _numeric_tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[-+]?\d+(?:[.,]\d+)?%?", text))


def _normalize(value: str) -> str:
    return value.strip().replace(",", ".").removesuffix(".0")
