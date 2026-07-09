# Gemma 3 12B compatibility spike -- hardware gate result

Date: 2026-07-09
Status: **Not attempted -- hardware gate failed before any load/inference step**

## HLD requirement

HLD SS13.5 / SS48.1 requires a compatibility spike before adopting Gemma 3
12B as the primary explanation model, covering checkpoint load, chat-template
behavior, structured output, memory headroom, LoRA smoke training, and
Gemma-terms acceptance recording. HLD SS22 sets the hardware precondition:
**at least 24 GB unified memory for 12B inference, 32 GB for QLoRA
training**, with Gemma 3 4B text-only retained as the lower-memory fallback
"if the 12B candidate fails hardware or release gates."

## Available hardware (this workspace, 2026-07-09)

- Mac mini, Apple M4, 10 CPU cores
- **16 GB unified memory** (`sysctl hw.memsize` = 17,179,869,184 bytes)
- macOS 26.5.1 arm64
- 48 GB free disk space

16 GB is below the 24 GB inference gate and well below the 32 GB QLoRA gate.
This is the same machine that produced the validated Gemma 3 4B benchmark
(`aeroroute-mlx/docs/NATIVE_BENCHMARK_2026-06-27.md` and
`aeroroute-mlx/docs/NATIVE_BENCHMARK_2026-07-09.md`), where the 4B checkpoint
used 2.2-3.4 GB peak resident memory -- comfortably inside 16 GB. A 4-bit
12B checkpoint is roughly 3x the parameter count of the 4B one; even before
accounting for KV cache, activation memory, and everything else this machine
needs to run concurrently, loading it would consume the large majority of
available unified memory with no headroom, and QLoRA training (which needs
gradient/optimizer state on top of that) is out of reach entirely on this
hardware.

## Prior attempt

`aeroroute-mlx/models/` already contains two incomplete local directories
from an earlier attempt to stage the 12B checkpoint:

- `gemma-3-text-12b-it-4bit/` -- 43 MB (config/tokenizer files only; the two
  `model-*-of-00002.safetensors` entries present by name are not real
  weights, a full 4-bit 12B checkpoint is on the order of 6-7 GB)
- `gemma-3-text-12b-it-4bit.partial-localdir-failed/` -- 93 MB, same
  incomplete pattern, named to record that the attempt failed

Neither directory contains a usable checkpoint. This is consistent with, and
was very likely caused by, the same hardware/environment constraints
recorded here.

## Decision

Per HLD SS22's own designed fallback, the compatibility spike for Gemma 3 12B
is **not attempted** on this machine. No load, generation, or memory
measurement was performed because the hardware precondition alone already
fails. Gemma 3 4B (`mlx-community/gemma-3-text-4b-it-4bit`, revision
`4f665a4c50ecfe4ecdc34056ab52fe3e3c4abf9e`) remains the validated model, per
the compatibility-and-fidelity gates it already passed. `configs/gemma3-12b-
qlora.yaml` is left in place as a forward-looking template for whenever this
project runs on qualifying hardware (>=24 GB for inference, >=32 GB for
QLoRA) -- it requires no code changes, only a capable machine.

## What would change this

Re-run this spike if either becomes true:

1. A Mac with >=24 GB unified memory becomes available for this workspace
   (the `MBPRO` and other machine folders visible elsewhere on this
   workspace's external drive may qualify -- not verified from this session).
2. A smaller or differently-quantized 12B-class checkpoint becomes available
   that plausibly fits in 16 GB with real headroom (would need its own ADR;
   HLD SS8.4 does not currently authorize deviating from the pinned
   `gemma-3-text-12b-it-4bit` checkpoint without one).

Until then, Phase 8 work on this hardware is scoped to strengthening the 4B
baseline (broader quality corpus, bake-off against similarly-sized
challengers such as Mistral 7B and Qwen3.5-9B) rather than promoting 12B.
