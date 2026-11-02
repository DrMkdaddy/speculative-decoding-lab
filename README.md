# speculative-decoding-lab

Measure when speculative decoding actually pays on one 16 GB GPU.

The setup is a Qwen3-4B target and a Qwen3-0.6B draft, both on the same
Vulkan device. The harness sweeps draft depth and confidence gating, records
acceptance at every draft position, and compares wall-clock throughput against
a bit-identical greedy baseline. The committed artifact includes the raw
per-prompt rows, so every number in the writeup can be recomputed.

## Why this exists

Speculative decoding is usually presented as a speedup. On a bandwidth-bound
consumer card it can also be a loss: the draft model costs a forward pass per
token, and the target must still verify every candidate. The break-even point
depends on the acceptance rate, the draft depth, and the ratio of draft to
target cost. This repo measures that boundary instead of assuming it.

## What is measured

For each configuration (`baseline`, `n2`, `n4`, `n6`, `n8`, `n6-p0.6`):

| Metric | Source |
|---|---|
| Decode throughput | `timings.predicted_per_second` from llama-server |
| Acceptance rate | `timings.draft_n_accepted / timings.draft_n` |
| Acceptance by position | Prometheus `spec_decode_num_accepted_tokens_per_pos_total` |
| Mean accepted length | `1 + accepted / verification_steps` |
| Prompt and first-token latency | `timings.prompt_ms`, `prompt_ms + predicted_ms / predicted_n` |
| VRAM | `rocm-smi` sampled while the server is resident |

All requests are greedy (`temperature=0`, `top_k=1`, fixed seed) so the target
output is the same with and without speculation. The harness asserts that the
generated token counts match and the output text is identical; a speedup that
changes the answer is not a speedup.

## Quick start

```bash
# Download the 0.6B draft (the 4B target is expected under ~/models)
python3 -m specdec fetch --config configs/qwen3-4b-0.6b.json

# Run the sweep (starts and stops llama-server for each configuration)
python3 -m specdec run --config configs/qwen3-4b-0.6b.json

# Render the tables
python3 -m specdec report results/spec_decode_*.json
```

Or:

```bash
./scripts/reproduce.sh
```

## Results

The committed run is in [`results/`](results/). The tables are generated from
the artifact:

<!-- RESULTS:START -->
Source: `results/spec_decode_qwen3-4b-instruct-2507__qwen3-0.6b_20261101T235537Z.json`

Target: Qwen3-4B-Instruct-2507 Q4_K_M. Draft: Qwen3-0.6B Q8_0.
One RX 9070 XT, Vulkan, 4k context, one slot, greedy, 128 tokens,
20 prompts x 3 runs per configuration.

| Config | Decode tok/s | Speedup vs baseline | Acceptance | Mean accepted len | Exact output |
|---|---:|---:|---:|---:|---:|
| baseline | 156.3 | 1.000x | - | - | - |
| n2 | 136.6 | 0.875x | 64.4% | 2.24 | **60/60** |
| n4 | 131.2 | 0.840x | 51.4% | 2.87 | 42/60 |
| n6 | 119.8 | 0.767x | 43.5% | 3.24 | 42/60 |
| n8 | 83.7 | 0.536x | 36.9% | 3.41 | 36/60 |
| n6-p0.6 | 135.7 | 0.869x | 76.2% | 3.57 | 36/60 |

### The result is negative, and that is the measurement

No speculative configuration beats the baseline on this target/draft pair.
The best **bit-exact** configuration is `n2`: 136.6 tok/s against 156.3,
a 12.5% loss, at 64.4% acceptance. The break-even fit says `n2` needs 81.1%
acceptance to reach 1.0x; the observed per-position curve falls from 71% at
the first draft position to 53% at the second.

Higher draft depths do not fix it: `n6` has a lower break-even acceptance
(62.2%) but its actual acceptance is only 43.5%, and its wall-clock speed is
0.767x. The confidence gate (`n6-p0.6`) raises acceptance to 76.2% and
recovers most of the speed (0.869x), but it is still slower, and it changes
the output on 24 of 60 prompts. That output change is the batch-composition
numerical effect documented in `docs/METHOD.md`; it is not a free win.

### Where it almost pays

| Class | Config | Speedup | Acceptance |
|---|---|---:|---:|
| json | n6-p0.6 | 1.026x | 82.0% |
| math | n6-p0.6 | 1.006x | 76.3% |
| json | n2 | 0.977x | 78.3% |
| code | n2 | 0.948x | 74.7% |
| prose | n2 | 0.747x | 47.2% |
| list | n2 | 0.743x | 47.0% |

The gated JSON/math slice is the only place speculation crosses 1.0x, and only
by 0.6-2.6%. The exact `n2` path stays below 1.0x in every class. That is the
decision boundary: on this card, a 0.6B draft against a 4B target is too
expensive unless acceptance is in the high 70s or better, which only the
structured-output prompts reach.

### Acceptance by draft position

| Config | p0 | p1 | p2 | p3 | p4 | p5 | p6 | p7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| n2 | 71% | 53% | - | - | - | - | - | - |
| n4 | 68% | 50% | 39% | 31% | - | - | - | - |
| n6 | 68% | 48% | 38% | 28% | 23% | 19% | - | - |
| n8 | 66% | 45% | 35% | 27% | 23% | 18% | 15% | 12% |
| n6-p0.6 | 87% | 56% | 40% | 29% | 24% | 20% | - | - |

### Break-even fit

| Config | Slope | Intercept | R² | Acceptance at 1.0x |
|---|---:|---:|---:|---:|
| n2 | 0.751 | 0.391 | 0.983 | 81.1% |
| n4 | 1.069 | 0.291 | 0.987 | 66.4% |
| n6 | 1.250 | 0.223 | 0.996 | 62.2% |
| n8 | 1.054 | 0.147 | 0.997 | 80.9% |
| n6-p0.6 | 1.293 | -0.117 | 0.471 | 86.3% |

### Router replay

Best fixed speculative config is `n2`. Decode time only.

| Policy | Total ms | vs always baseline | Spec requests |
|---|---:|---:|---:|
| Always baseline | 48,765 | +0.0% | - |
| Always spec (n2) | 56,832 | +16.5% | - |
| Oracle per prompt | 47,722 | -2.1% | 17 |
| Per-class, leave-one-out | 48,951 | +0.4% | 15 |
| Online EWMA | 49,208 | +0.9% | 17 |

The per-class router chooses `n6-p0.6` for 15 of 60 requests, but the class
gains are too small to overcome the losses elsewhere; the policy is 0.4%
slower than always-baseline on the replay. The online EWMA is 0.9% slower.
The oracle, which sees each prompt's outcome before choosing, could save 2.1%.
That gap is the honest cost of deciding before the request is served.

### What the harness does with output divergence

`n2` is byte-identical to the baseline on all 60 outputs. At `n4`, `n6`,
`n8`, and `n6-p0.6`, the verification batch changes the floating-point
composition on this Vulkan stack and 18-24 of 60 outputs differ. The artifact
records exact-match counts and mean common-prefix agreement. The speed numbers
for those rows are still measured, but they are not presented as a drop-in
replacement that preserves the answer.
<!-- RESULTS:END -->

## The router

A fixed global speculation setting is the wrong policy when prompt classes
have different acceptance. `specdec/router.py` implements an online policy:

1. Classify the prompt from cheap features (class tag, token count, shape).
2. Keep an EWMA of measured speedup per class.
3. Enable speculation only while the class's EWMA is above 1.0.
4. Fall back to the baseline path when the estimate is below break-even.

The committed analysis evaluates the policy by leave-one-prompt-out replay on
the measured rows. The decision boundary is the acceptance rate where the
measured speedup crosses 1.0; it is reported with the artifact. This is a
measurement of a policy, not a claim that two models run concurrently on one
card.

## Caveats

- One target, one draft, one GPU. The acceptance rate is model-pair specific.
- The baseline and speculative servers are run separately, not concurrently.
  Wall-clock comparisons are therefore clean, but the router's online
  evaluation is a replay of measured latencies, not a live two-server test.
- The draft is Q8_0 and the target Q4_K_M, which is the realistic asymmetric
  pairing; a different draft quant changes the cost side of the boundary.
- This harness measures greedy decoding. Sampling changes acceptance.

## Layout

```
specdec/
  cli.py       fetch / run / report / route
  server.py    llama-server lifecycle
  client.py    completion requests and timings
  metrics.py   Prometheus parser
  analyze.py   acceptance, break-even, router replay
  router.py    online acceptance/speed policy
configs/       model pair and sweep grid
prompts/       deterministic prompt set with class labels
scripts/       reproduce.sh
tests/         parser and analysis tests, no GPU required
results/       committed JSON artifact and logs
```

## License

MIT.
