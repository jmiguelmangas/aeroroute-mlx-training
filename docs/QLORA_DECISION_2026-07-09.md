# Phase 8B (QLoRA) decision: not attempted, not justified by current evidence

Date: 2026-07-09
Status: **Deferred -- closes checkpoint "Required closure sequence" step 2
("Keep Phase 8B unpromoted unless [Phase 8] baselines justify training")
with a documented decision rather than silence.**

## What Phase 8 baselines actually showed

Per HLD SS31.1, Phase 8B "starts only after the prompt-only service and
evaluation corpus are working," and its own work items (6-7) are to record
prompt-only/few-shot baselines and run a smoke QLoRA experiment only once
those baselines exist. That evidence now exists:

- `aeroroute-mlx-training/docs/COMPATIBILITY_12B_2026-07-09.md`: Gemma 3
  12B fails the HLD SS22 hardware gate on this machine (16 GB < 24 GB
  required for inference, < 32 GB for QLoRA); not attempted.
- `aeroroute-mlx/docs/QUALITY_CORPUS_2026-07-09.md`: the prompt-only Gemma
  3 4B baseline scores **100% pass rate** (23/23 evaluated cases) on a
  24-case corpus covering schema validity, numeric fidelity, and
  operational-claim safety -- the exact dimensions HLD SS13.9's gates
  measure.
- `aeroroute-mlx-training/docs/BAKEOFF_2026-07-09.md`: neither challenger
  model beats this baseline. Mistral 7B Instruct v0.3 matches quality at
  ~1.5-1.8x the latency; Qwen3-8B fails the structured-output contract on
  more than half the corpus.

## Why this does not justify building the QLoRA pipeline now

QLoRA's stated purpose (HLD SS13.7) is narrow: teach the model to "produce
the versioned explanation JSON schema; identify the supplied winning route
and objective profile; reproduce numeric facts without mutation; describe
signed deltas and trade-offs correctly; use concise aviation-aware
terminology; include supplied assumptions, data-quality warnings, and
non-operational limitations." The prompt-only 4B baseline is already doing
all of this at 100% measured pass rate, with no fallback triggered, on a
corpus deliberately spanning 6 aircraft types, 3 profiles, and both
original and newly added airports. There is no visible quality gap for an
adapter to close against this evidence -- fine-tuning a model that is
already at ceiling on every metric this project currently measures has
low expected value.

Attempting it anyway would require substantial net-new engineering, not
just a training run:

- `aeroroute-mlx-training/src/aeroroute_mlx_training/` has no `training/`,
  `publishing/`, or `reporting/` modules yet -- only dataset-record shape,
  grouped splitting, and evaluation-metric scoring exist. The actual LoRA
  fine-tuning loop, checkpoint save/reload, adapter manifest generation,
  and promotion-gate wiring (HLD SS31.1 items 8-12) do not exist.
- No training dataset exists. HLD SS13.8 targets 3,000-5,000 examples
  (500-1,000 held out, 200+ manually reviewed gold) split by route/
  aircraft/weather family, not random rows. Even a reduced smoke-scale run
  (HLD SS13.7.1: "50-100 step smoke experiment before every full training
  configuration") needs a real generated-and-reviewed set, which does not
  exist -- `configs/gemma3-4b-smoke.yaml` is a placeholder stub missing
  `base_revision`, `max_seq_length`, `dataset_version`, and every other
  field its 12B sibling config already has filled in.

## Decision

Do not build the QLoRA training pipeline or attempt a smoke run at this
time. The prompt-only Gemma 3 4B baseline remains the model used by
`aeroroute-mlx`. This closes the checkpoint's Phase 8B closure condition
with an evidence-based "not justified" rather than leaving it silently
unaddressed.

## What would change this

Revisit if any of the following becomes true:

1. A quality corpus (this one or a larger successor) surfaces a real,
   reproducible gap in the 4B baseline's explanations -- something worth
   fine-tuning away. Nothing in the current 24-case corpus does.
2. Hardware qualifying for the 12B gate becomes available (HLD SS22:
   >=24 GB inference, >=32 GB QLoRA) and a future 12B compatibility spike
   passes -- QLoRA against a stronger base model would be a different
   value proposition than fine-tuning an already-ceiling 4B baseline.
3. A new, concrete product requirement needs behavior the prompt-only
   baseline cannot express (e.g. a longer or structurally different
   explanation format) that prompting alone cannot reliably satisfy.

Until then, `aeroroute-mlx-training`'s `training/`, `publishing/`, and
`reporting/` modules remain unbuilt by design, not by oversight.
