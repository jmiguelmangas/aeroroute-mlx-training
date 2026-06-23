"""Deterministic synthetic explanation records from frozen optimizer results."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ExplanationRecord:
    record_id: str
    group_id: str
    contract_version: str
    facts: dict[str, object]
    target: str


def generate_record(run: dict[str, object]) -> ExplanationRecord:
    origin = str(run["origin_icao"])
    destination = str(run["destination_icao"])
    aircraft_type = str(run["aircraft_type"])
    facts = dict(run["facts"])
    canonical = json.dumps(run, sort_keys=True, separators=(",", ":"))
    record_id = hashlib.sha256(canonical.encode()).hexdigest()
    group_id = f"{origin}-{destination}-{aircraft_type}"
    target = _render_target(origin, destination, facts)
    return ExplanationRecord(record_id, group_id, "1.0.0", facts, target)


def serialize_record(record: ExplanationRecord) -> str:
    return json.dumps(asdict(record), sort_keys=True, separators=(",", ":"))


def _render_target(
    origin: str, destination: str, facts: dict[str, object]
) -> str:
    return (
        f"The synthetic route from {origin} to {destination} uses "
        f"{facts['fuel_kg']} kg of estimated fuel and takes {facts['time_min']} minutes."
    )
