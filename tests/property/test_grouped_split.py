from aeroroute_mlx_training.dataset_generation.records import generate_record
from aeroroute_mlx_training.splitting.grouped import split_by_group


def _record(origin: str, destination: str):
    return generate_record(
        {
            "origin_icao": origin,
            "destination_icao": destination,
            "aircraft_type": "A320",
            "facts": {"fuel_kg": 1, "time_min": 1},
        }
    )


def test_same_route_group_never_leaks_across_splits() -> None:
    records = (
        _record("LEMD", "KJFK"),
        _record("LEMD", "KJFK"),
        _record("EGLL", "OMDB"),
    )
    splits = split_by_group(records)
    locations = {
        record.record_id: name
        for name, split in (
            ("train", splits.train),
            ("validation", splits.validation),
            ("test", splits.test),
        )
        for record in split
    }

    assert locations[records[0].record_id] == locations[records[1].record_id]
