from pathlib import Path

from aeroroute_mlx_training.evaluation.adapter import (
    AdapterPrediction,
    _generate_completion,
    evaluate_predictions,
    evaluate_template_baseline,
)
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


def test_evaluation_requires_every_allowed_numeric_value() -> None:
    report = evaluate(
        (
            EvaluationCase(
                "missing-number",
                "minimum_fuel",
                ("18000", "420"),
                "The selected synthetic route uses 18000 kg.",
                "minimum_fuel",
            ),
        )
    )
    assert report.numeric_fidelity == 0.0


def test_adapter_evaluation_is_not_promotion_eligible_without_winner_metric() -> (
    None
):
    report = evaluate_predictions(
        (
            AdapterPrediction(
                "held-out-1",
                ("18000", "420"),
                "The route uses 18000 kg and takes 420 minutes.",
            ),
        )
    )
    assert report.numeric_fidelity == 1.0
    assert report.promotion_eligible is False
    assert report.promotion_blocker == "winner_correctness_not_evaluated"


def test_adapter_generation_uses_only_supported_greedy_options() -> None:
    captured: dict[str, object] = {}

    def fake_generate(*args: object, **kwargs: object) -> str:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "completion"

    assert (
        _generate_completion(fake_generate, "model", "tokenizer", "prompt", 80)
        == "completion"
    )
    assert captured["kwargs"] == {
        "prompt": "prompt",
        "max_tokens": 80,
        "verbose": False,
    }


def test_template_baseline_scores_the_held_out_targets(tmp_path: Path) -> None:
    (tmp_path / "test.jsonl").write_text(
        '{"messages":[{"role":"system","content":"x"},'
        '{"role":"user","content":"{\\"fuel_kg\\": 18000, \\"time_min\\": 420}"},'
        '{"role":"assistant","content":"Uses 18000 kg in 420 minutes."}]}\n'
    )
    report = evaluate_template_baseline(tmp_path)
    assert report.numeric_fidelity == 1.0
    assert report.unsupported_claim_rate == 0.0
