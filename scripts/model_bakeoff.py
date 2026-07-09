"""Challenger model bake-off against the same corpus as the Gemma 3 4B baseline.

HLD SS13.5's model portfolio names Mistral 7B Instruct v0.3 and "Qwen3.5-9B"
as measured challengers, never runtime dependencies. This script runs the
identical 24-case corpus used in
``aeroroute-mlx/scripts/quality_corpus.py`` (same routes, aircraft,
profiles -- duplicated verbatim here for a fair comparison) against a given
challenger checkpoint, using the same prompt template and validators as the
production explanation service (duplicated from ``aeroroute-mlx/src/
aeroroute_mlx/{prompt_builder,validator}.py`` -- this repo must not become a
runtime dependency of aeroroute-mlx, so the small pure logic is copied
rather than imported).

Deliberately does NOT use aeroroute-mlx's ``ModelManifest``/`MlxLmProvider``:
those are intentionally locked to the approved ``gemma3_text`` production
architecture (see ``ModelManifest.validate()``), and challenger models here
are evaluation-only, never promoted to that serving path.

Note on "Qwen3.5-9B": the exact model HLD names under that identifier
(``mlx-community/Qwen3.5-9B-MLX-4bit`` and ``-4bit``) is a vision-language
model requiring ``mlx_vlm``, not the text-only ``mlx_lm`` architecture this
project is scoped to (see HLD SS13.5: "vision inputs and mlx-vlm are outside
this project"). Substituted with ``mlx-community/Qwen3-8B-4bit`` (text-only,
same family, closest available parameter count) -- see
``docs/BAKEOFF_2026-07-09.md`` for the full rationale.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import resource
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

API_BASE_URL = "http://127.0.0.1:8000"

# Identical to aeroroute-mlx/scripts/quality_corpus.py CORPUS_ROUTES.
CORPUS_ROUTES: tuple[tuple[str, str, str, str], ...] = (
    ("LEMD", "LFPG", "A320", "balanced"),
    ("EGLL", "EDDF", "A320", "minimum_time"),
    ("EHAM", "LOWW", "B738", "minimum_fuel"),
    ("LEMD", "EDDM", "A320", "balanced"),
    ("LFPG", "LIMC", "B738", "minimum_time"),
    ("EGLL", "EBBR", "A320", "minimum_fuel"),
    ("LEMD", "KJFK", "B77W", "balanced"),
    ("KJFK", "LEMD", "B788", "minimum_fuel"),
    ("EDDF", "KATL", "B77W", "minimum_time"),
    ("EGLL", "KBOS", "B788", "balanced"),
    ("LEMD", "MMMX", "A359", "minimum_fuel"),
    ("KJFK", "CYUL", "A320", "minimum_time"),
    ("OMDB", "LEMD", "A359", "balanced"),
    ("OMDB", "SKBO", "B77W", "minimum_time"),
    ("OTHH", "HKJK", "B788", "minimum_fuel"),
    ("HECA", "FAOR", "A359", "balanced"),
    ("OMDB", "LTFM", "A320", "minimum_time"),
    ("RJAA", "KSFO", "B788", "minimum_fuel"),
    ("RJTT", "ZBAA", "A359", "balanced"),
    ("RKSI", "WSSS", "B738", "minimum_time"),
    ("VHHH", "WMKK", "A320", "minimum_fuel"),
    ("WSSS", "VTBS", "A320", "balanced"),
    ("YSSY", "NZCH", "B738", "minimum_time"),
    ("SBGR", "SBGL", "A320", "minimum_fuel"),
)

_NUMERIC_TOKEN = re.compile(r"[-+]?\d+(?:[.,]\d+)?%?")
_BANNED_OPERATIONAL_CLAIMS = (
    "atc clearance",
    "cleared for",
    "guarantees safety",
    "operational flight plan",
    "safe route",
)


@dataclass(frozen=True, slots=True)
class ExplanationFacts:
    origin_icao: str
    destination_icao: str
    profile: str
    distance_m: float
    time_s: float
    fuel_kg: float
    data_degraded: bool = False
    baseline_time_s: float | None = None
    baseline_fuel_kg: float | None = None


DEGRADED_CODES = {
    "WEATHER_FALLBACK",
    "WEATHER_STALE",
    "WEATHER_STILL_AIR",
    "FUEL_NOT_CONVERGED",
}


def facts_from_optimization(doc: dict[str, Any]) -> ExplanationFacts:
    request = doc["request"]
    winner = doc["winner"]
    baseline = doc.get("baseline")
    return ExplanationFacts(
        origin_icao=request["origin_icao"],
        destination_icao=request["destination_icao"],
        profile=request["profile"],
        distance_m=winner["distance_m"],
        time_s=winner["time_s"],
        fuel_kg=winner["fuel_kg"],
        data_degraded=any(
            flag["code"] in DEGRADED_CODES for flag in doc.get("data_quality", [])
        ),
        baseline_time_s=(baseline["time_s"] if baseline else None),
        baseline_fuel_kg=(baseline["fuel_kg"] if baseline else None),
    )


def render_deterministic_explanation(facts: ExplanationFacts) -> str:
    distance_km = facts.distance_m / 1_000
    time_minutes = facts.time_s / 60
    comparison = _comparison_text(facts)
    return (
        f"For the {facts.profile} profile, the selected synthetic trajectory "
        f"from {facts.origin_icao} to {facts.destination_icao} covers "
        f"{distance_km:.0f} km, takes an estimated {time_minutes:.0f} minutes, "
        f"and uses {facts.fuel_kg:.0f} kg of modeled trip fuel. {comparison}"
        "This is an educational trajectory-efficiency estimate, not "
        "operational flight-planning advice."
    )


def allowed_numeric_values(facts: ExplanationFacts) -> list[str]:
    values = {
        f"{facts.distance_m / 1_000:.0f}",
        f"{facts.time_s / 60:.0f}",
        f"{facts.fuel_kg:.0f}",
    }
    if facts.baseline_time_s is not None:
        values.add(f"{abs(facts.time_s - facts.baseline_time_s) / 60:.0f}")
    if facts.baseline_fuel_kg is not None:
        values.add(f"{abs(facts.fuel_kg - facts.baseline_fuel_kg):.0f}")
    return sorted(values)


def _comparison_text(facts: ExplanationFacts) -> str:
    if facts.baseline_fuel_kg is None or facts.baseline_time_s is None:
        return ""
    fuel_delta = facts.fuel_kg - facts.baseline_fuel_kg
    time_delta_minutes = (facts.time_s - facts.baseline_time_s) / 60
    fuel_text = _delta_text(fuel_delta, "kg of modeled trip fuel")
    time_text = _delta_text(time_delta_minutes, "minutes")
    return f"Compared with the baseline, it {fuel_text} and {time_text}. "


def _delta_text(delta: float, unit: str) -> str:
    rounded = abs(delta)
    if rounded < 0.5:
        return f"has a negligible difference in {unit}"
    if delta < 0:
        return f"saves {rounded:.0f} {unit}"
    return f"uses {rounded:.0f} more {unit}"


def build_prompt(summary: str, allowed_values: list[str]) -> str:
    allowed = ", ".join(allowed_values) or "no numeric claims"
    return (
        "Return exactly one JSON object with this schema: "
        '{"text":"brief explanation"}. Explain only the supplied '
        "deterministic result. Do not calculate, rank, add facts, make safety "
        "claims, or mention ATC clearance. Allowed numeric values: "
        f"{allowed}.\n\nDeterministic summary:\n{summary}"
    )


def parse_structured_text(raw: str) -> str:
    candidate = raw.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        candidate = "\n".join(lines[1:-1]).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model output is not a JSON object")
    payload = json.loads(candidate[start : end + 1])
    if set(payload) != {"text"} or not isinstance(payload["text"], str):
        raise ValueError("model output does not match the explanation schema")
    text = payload["text"].strip()
    if not text:
        raise ValueError("model explanation is empty")
    return text


def validate_numeric_claims(text: str, allowed_values: list[str]) -> bool:
    allowed = {_normalize(value) for value in allowed_values}
    return all(
        _normalize(token) in allowed for token in _NUMERIC_TOKEN.findall(text)
    )


def validate_operational_claims(text: str) -> bool:
    normalized = text.casefold()
    return not any(claim in normalized for claim in _BANNED_OPERATIONAL_CLAIMS)


def _normalize(value: str) -> str:
    return value.strip().replace(",", ".").removesuffix(".0")


def fetch_optimization(
    origin: str, destination: str, aircraft: str, profile: str
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "origin_icao": origin,
            "destination_icao": destination,
            "aircraft_type": aircraft,
            "profile": profile,
        }
    ).encode()
    request = urllib.request.Request(
        f"{API_BASE_URL}/api/v1/optimizations",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(quantile * (len(ordered) - 1))))
    return ordered[index]


async def generate_one(model: Any, tokenizer: Any, prompt: str, timeout_s: float) -> str:
    from mlx_lm import generate as mlx_generate

    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return await asyncio.wait_for(
        asyncio.to_thread(
            mlx_generate,
            model,
            tokenizer,
            prompt=chat_prompt,
            max_tokens=180,
            verbose=False,
        ),
        timeout=timeout_s,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-path", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from mlx_lm import load as mlx_load

    model, tokenizer = mlx_load(str(args.local_path.resolve()), lazy=False)

    cases: list[dict[str, Any]] = []
    for index, (origin, destination, aircraft, profile) in enumerate(
        CORPUS_ROUTES, start=1
    ):
        case_id = f"{origin}-{destination}-{aircraft}-{profile}"
        print(f"[{args.label}] [{index}/{len(CORPUS_ROUTES)}] {case_id} ...", flush=True)
        try:
            doc = fetch_optimization(origin, destination, aircraft, profile)
        except Exception as error:  # noqa: BLE001
            cases.append(
                {"case_id": case_id, "api_error": str(error), "passed": False}
            )
            continue
        facts = facts_from_optimization(doc)
        summary = render_deterministic_explanation(facts)
        numeric_values = allowed_numeric_values(facts)
        prompt = build_prompt(summary, numeric_values)

        started = time.perf_counter()
        passed = False
        text = ""
        error_reason = None
        try:
            raw = asyncio.run(generate_one(model, tokenizer, prompt, timeout_s=60))
            text = parse_structured_text(raw)
            if validate_numeric_claims(
                text, numeric_values
            ) and validate_operational_claims(text):
                passed = True
            else:
                error_reason = "validation_failed"
        except Exception as error:  # noqa: BLE001
            error_reason = f"{type(error).__name__}: {error}"
        elapsed_s = time.perf_counter() - started

        cases.append(
            {
                "case_id": case_id,
                "data_degraded": facts.data_degraded,
                "latency_s": round(elapsed_s, 3),
                "passed": passed,
                "error_reason": error_reason,
                "text": text,
            }
        )

    evaluated = [case for case in cases if "latency_s" in case]
    passed_cases = [case for case in evaluated if case["passed"]]
    latencies = [case["latency_s"] for case in evaluated]
    report = {
        "label": args.label,
        "repo_id": args.repo_id,
        "revision": args.revision,
        "total_cases": len(cases),
        "api_errors": len(cases) - len(evaluated),
        "evaluated_cases": len(evaluated),
        "passed_cases": len(passed_cases),
        "pass_rate": round(len(passed_cases) / len(evaluated), 4) if evaluated else 0.0,
        "latency_p50_s": round(percentile(latencies, 0.5), 3) if latencies else None,
        "latency_p95_s": round(percentile(latencies, 0.95), 3) if latencies else None,
        "peak_resident_memory_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 1
        ),
        "cases": cases,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
