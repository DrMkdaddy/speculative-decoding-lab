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
Run `python3 -m specdec report results/spec_decode_*.json` after a sweep.
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
