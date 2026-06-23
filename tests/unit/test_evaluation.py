from aeroroute_mlx_training.evaluation.metrics import EvaluationCase, evaluate
from aeroroute_mlx_training.evaluation.promotion import decide_promotion


def test_evaluation_detects_unsupported_numeric_claims() -> None:
    report = evaluate(
        (
            EvaluationCase(
                "one", "fuel", ("1200", "90"), "Fuel 1200 in 90 min", "fuel"
            ),
            EvaluationCase("two", "time", ("500",), "Fuel 999", "time"),
        )
    )

    assert report.winner_correctness == 1.0
    assert report.numeric_fidelity == 0.5
    assert report.unsupported_claim_rate == 0.5


def test_promotion_fails_on_critical_fidelity_regression() -> None:
    baseline = evaluate((EvaluationCase("one", "fuel", ("1",), "1", "fuel"),))
    candidate = evaluate((EvaluationCase("one", "fuel", ("1",), "2", "fuel"),))

    decision = decide_promotion(baseline, candidate)

    assert not decision.promoted
    assert "numeric_fidelity_below_gate" in decision.reasons
