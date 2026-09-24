# Benchmark policy — what a number must carry before it is published

> **Ported from `3spark-qwen-3.8-27b-dense/docs/BENCHMARK-POLICY.md` on
> 2026-09-03**, itself ported from `3spark-dsv4`. This repository had no copy.
> The SGLang TP=2/TP=3 campaign in the dense-model repo violated four of the
> five hard requirements below on the same day the standard existed elsewhere
> — the rules are engine-agnostic but a standard that lives in one repo
> protects no other. Only the fifth-rate `bench-miaai.py` harness that all
> three Qwen/DSv4 repos share was already compliant with requirement 2
> (`min_tokens==max_tokens`, `ignore_eos`, and a hard `WindowCollapse` abort);
> requirements 1, 3, 4 and 5 still need explicit wiring per repo.

Every invalidated result in the sibling repositories failed one of the rules
below, and each rule exists because we published something wrong. This page is
the standard for any new benchmark in this repo.

## The five hard requirements

### 1. A fabric gate must run, and its artifact must be committed

**Rule:** `scripts/fabric_gate.sh` runs before every measurement arm, with the
engine in the state it will be measured in, and its JSON artifact is committed
next to the results.

**Status in this repo: NOT YET PORTED.** `fabric_gate.sh` depends on
`common.sh`, a `configs/*.env` node list, and `agbench.py`, none of which exist
here yet. Until it is ported, any sweep in this repo is `fabric_gate: ABSENT`
in `results/index.yaml` — real but unverified against a degraded-fabric false
reading. See the dense-model repo's `docs/BENCHMARK-POLICY.md` for why this
matters: a node ran at ~15% of healthy collective bandwidth for four days with
every error counter reading zero.

### 2. The decode window must be long enough, and verified

**Rule:** force the output length (`min_tokens == max_tokens` plus
`ignore_eos`), use **at least 256 tokens**, and **assert
`completion_tokens == max_tokens` per rep**. The run must fail, not warn, when
the window collapses.

**Status in this repo: COMPLIANT.** `bench-miaai.py` (shared with the other
two repos) already sets `min_tokens`/`ignore_eos` and raises `WindowCollapse`
on a short completion. `qwen-next-sweep.sh` calls `--output-tokens` explicitly
— confirm it is passed and >= 256 for every published sweep, and that the
value is recorded in `results/index.yaml`.

### 3. Publish the spread, never the median alone

**Rule:** every result publishes sorted per-rep values, not just a median, and
commits the raw per-rep file.

**Status in this repo: PARTIAL.** `qwen-next-sweep.sh`'s `rows.tsv` currently
has no min/max spread columns (only `median_decode_tok_s`/`agg_tok_s`/
`ttft_ms`). Raw `bench-c*.log` files are committed, so spread is recoverable
by hand, but the harness should compute and publish it directly — see the
dense-model repo's sed-bug history for what happens when spread is left to a
one-off recomputation.

### 4. Config comes from the live process

**Rule:** capture engine config with `ps -eo args` (or `docker inspect`) on the
running engine and the KV pool size from that boot's log — **not** from a
config/recipe YAML, which may not be what actually launched.

**Why it matters here specifically:** this repo's configs are recipe YAMLs
consumed by `run-recipe.py`, and the boot scripts also pass CLI overrides
(`--gpu-memory-utilization`) that can silently diverge from the YAML default.
The working tree has already shown this: `qwen-next-boot-tp3.sh` was edited to
`0.82` while `qwen-next-boot-tp2.sh` still hardcodes `0.80` — a result bundle
must record what the container actually ran, not what the yaml says.

### 5. The engine must be exclusively ours for the duration

**Rule:** before the first measured request, assert
`vllm:num_requests_running == 0` and `vllm:num_requests_waiting == 0`. Record
`vllm:request_success_total` at start and end; the delta must equal the number
of requests the harness itself issued. A larger delta means another client was
served during the measurement window and the run is **void**, not slow.

**Status in this repo: NOT YET WIRED.** `scripts/exclusivity.py` has been
ported from the dense-model repo (itself from DSv4) — it is engine-agnostic
vLLM `/metrics` scraping, and this repo's engine is also
`eugr/spark-vllm-b12x`, so no metric-name translation is needed. It is not yet
called from `qwen-next-sweep.sh`. Wire it in before trusting a comparison
across concurrency levels sharing one boot.

## Cold vs warm

Cold and warm are different measurements and must never be mixed. Warm at
least 2 requests per prompt shape before measuring (`qwen-next-sweep.sh`
already does `c in {1,4}` warmup, discarded). A comparison is invalid if one
side is warm and the other cold.

## Comparing against other repositories

Numbers from another repository are **not** comparable to ours unless the
harness, output-window length, prompt, warm-up state, and `max_num_seqs` all
match. This is why the 3-node TP=3 result in `results/index.yaml`
(`20260902-tp3-mnbt8192`) has no TP=2 counterpart yet in this repo — there is
no matched arm to compare it against until a TP=2 sweep is run with the same
harness, prompt, and window.

**Cross-repo material is valuable for config and methodology. It is not
valuable for numbers.**
