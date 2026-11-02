# Method

The unit of measurement is one completion request with a fixed prompt, greedy
decoding, and a fixed output length. That makes the baseline and speculative
paths comparable without hiding a sampling difference inside the comparison.

## Models

- Targets: `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` and
  `Qwen3-8B-Q4_K_M.gguf`
- Draft: `Qwen3-0.6B-Q8_0.gguf`
- Runtime: llama.cpp `0.7.0-dev`, build 786, commit `fcfb89b`
- Backend: Vulkan, all layers on one AMD RX 9070 XT (RDNA4, gfx1201)
- Context: 4096 tokens, one server slot
- KV cache: f16

The 4B sweep uses three repetitions per prompt; the 8B sweep uses two. Both
use the same 20 prompts and the same greedy request shape. The target-size
comparison is the point: the same draft is a large fraction of the 4B step and
a smaller fraction of the 8B step.

## Request shape

- Raw `/completion`, not a chat template.
- `temperature=0`, `top_k=1`, `top_p=1`, `min_p=0`
- `cache_prompt=false`
- `ignore_eos=true`, `n_predict=128`, seed fixed per run
- 20 prompts in five classes: prose, code, JSON, math, and list
- 3 repetitions per prompt per configuration

The harness compares every speculative output to the baseline output for the
same prompt and run. Exact-match counts and mean common-prefix agreement are
recorded per configuration. `n2` is byte-identical on all 60 outputs; higher
draft depths diverge on some prompts because the verification batch changes the
floating-point composition on this Vulkan stack. Those rows are kept, but their
speedups are not presented as answer-preserving replacements.

## Configurations

| Name | Draft depth | Confidence gate |
|---|---:|---:|
| baseline | - | - |
| n2 | 2 | none |
| n4 | 4 | none |
| n6 | 6 | none |
| n8 | 8 | none |
| n6-p0.6 | 6 | `p_min=0.6` |

## Acceptance

Per request, llama-server returns `draft_n` and `draft_n_accepted` in the
completion timings. The acceptance rate is `accepted / drafted`.

The per-position curve comes from the Prometheus counter
`llamacpp:spec_decode_num_accepted_tokens_per_pos_total`. The harness
snapshots the counter before and after each request. The denominator is the
number of verification steps, `spec_decode_num_drafts_total`, following the
same convention llama.cpp uses in its own log line. A position that was never
drafted because the previous token was rejected is therefore not counted as a
failure for that position.

## Break-even

For each configuration, the analysis fits a line

```
measured speedup = a + b * acceptance
```

across the per-prompt rows. The crossing at speedup 1.0 is the acceptance rate
needed for that draft depth to pay for itself on this target/draft pair. The
fit is descriptive, not causal: acceptance and speedup are both measured on
the same request, and the relation can be nonlinear at the edges. The R² and
the sample count are reported with the crossing.

## Router replay

The router is evaluated two ways on the measured rows:

1. **Per-class leave-one-prompt-out.** For a prompt, the class mean speedup is
   computed from the other prompt IDs in that class. The prompt is routed to
   the speculative path if that mean is above 1.0, otherwise to the baseline.
   The scored prompt never contributes to its own decision.
2. **Online EWMA.** A fixed shuffled order; each observation updates a
   per-class EWMA with `alpha=0.3`; the next request is routed using the EWMA
   before the observation. This is the order an online router would see.

The replay compares total decode milliseconds against always-baseline,
always-spec, and a per-prompt oracle. It is a replay, not a live two-server
test: running both servers concurrently on one card would contend for the same
GPU and change the latencies being compared.

## What is not claimed

- No claim about a different target/draft pair beyond these two targets.
- No claim about sampling; all measurements are greedy.
- No claim that the router improves throughput on a live deployment. The
  replay measures the policy on this trace.
- No claim that the draft model is free. Its VRAM and per-token cost are part
  of the result.
- The 4B result is not a speedup. Only the 8B `n2` result is presented as a
  drop-in speedup, because it is the only configuration that is both faster
  and byte-identical.
