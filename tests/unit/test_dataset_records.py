from aeroroute_mlx_training.dataset_generation.records import generate_record
from aeroroute_mlx_training.dataset_validation.validation import validate_record


def _run() -> dict[str, object]:
    return {
        "origin_icao": "LEMD",
        "destination_icao": "KJFK",
        "aircraft_type": "A320",
        "facts": {"fuel_kg": 18000, "time_min": 420},
    }


def test_generation_is_deterministic_and_valid() -> None:
    first = generate_record(_run())
    second = generate_record(_run())

    assert first == second
    validate_record(first)


def test_validator_rejects_target_with_unsupported_numeric_claim() -> None:
    record = generate_record(_run())
    invalid = record.__class__(
        record.record_id,
        record.group_id,
        record.contract_version,
        record.facts,
        "Fuel is 999 kg.",
    )

    try:
        validate_record(invalid)
    except ValueError as error:
        assert "fuel_kg" in str(error)
    else:
        raise AssertionError("invalid numeric claim must be rejected")
