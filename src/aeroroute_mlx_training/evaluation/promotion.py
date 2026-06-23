"""Promotion gates shared by prompt-only, few-shot, and QLoRA comparisons."""

from dataclasses import dataclass

from aeroroute_mlx_training.evaluation.metrics import EvaluationReport


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    promoted: bool
    reasons: tuple[str, ...]


def decide_promotion(
    baseline: EvaluationReport,
    candidate: EvaluationReport,
    minimum_numeric_fidelity: float = 0.98,
    maximum_unsupported_claim_rate: float = 0.0,
) -> PromotionDecision:
    reasons: list[str] = []
    if candidate.numeric_fidelity < minimum_numeric_fidelity:
        reasons.append("numeric_fidelity_below_gate")
    if candidate.unsupported_claim_rate > maximum_unsupported_claim_rate:
        reasons.append("unsupported_claim_rate_above_gate")
    if candidate.winner_correctness < baseline.winner_correctness:
        reasons.append("winner_correctness_regressed")
    return PromotionDecision(not reasons, tuple(reasons))
