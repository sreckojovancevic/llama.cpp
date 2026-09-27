# WeightProvider fork plan

This is a personal fork of llama.cpp (upstream: ggml-org/llama.cpp).
Working branch: `weight-provider-experiment`.
**Never open pull requests against ggml-org/llama.cpp.** Push only to this fork.

## Goal

Better placement of MoE experts across devices and memory tiers:
hot experts on GPU, warm experts computed on CPU from RAM, cold experts left on
disk (mmap). Remote RPC backends are just another compute device.

Two concepts, kept separate:

- **Placement** - on which device an expert is *computed* (GPU, CPU, RPC device).
  Weights live there; only activations move.
- **Residency** - in which local memory tier an expert's weights currently *live*
  (VRAM, RAM, NVMe/SSD). Weights move toward compute (cache, prefetch, evict).

The project goes in phases. **Do not start a later phase before the earlier one
is done and measured.**

- **Phase 0 (now):** measure. Trace which experts the router selects, per layer,
  and analyze the distribution. This decides whether Phase 1 is worth doing.
- **Phase 1:** static, profile-guided placement (hot/cold split per layer).
- **Phase 2 (maybe never):** dynamic residency with prefetch/eviction. Only if
  Phase 0 data shows the hot set shifts significantly between workloads.

## Current state of the branch (commit 20060ac09)

1. `ggml/src/ggml-cuda/ggml-cuda.cu`: logging inside `ggml_cuda_mul_mat_id`.
   Problems: no layer index; counted 3x per layer (gate/up/down); CUDA only
   (CPU experts invisible); `tokens_per_expert` only exists on the host-sync
   path, not on the MMQ fast path; `GGML_LOG_INFO` per call floods the log.
   **Action: revert this hunk.** Tracing moves to graph level (Task 1).
2. `src/llama-arch.cpp`, `src/llama-model.h`, `src/models/llama.cpp`: loader for
   per-expert split tensors (`ffn_*_exp` vectors). Problem: the graph
   (`build_moe_ffn`) still uses the merged `ffn_*_exps` tensors, so a split-tensor
   model loads and then crashes on first forward. Also, one MUL_MAT per expert is
   the wrong granularity for compute (hundreds of graph nodes per layer, loses
   the batched MUL_MAT_ID kernel).
   **Action: revert these hunks** (or leave them untouched and unused; do not
   build on them). Phase 1 uses a different design, see below.

Status: both hunks reverted (commit "Revert experimental split MoE expert loading").

## Environment constraints (cloud sandbox)

- Linux, **no GPU**. Build CPU only:
  ```
  cmake -B build -DGGML_CUDA=OFF -DGGML_METAL=OFF -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j
  ```
- Run existing tests after changes: `ctest --test-dir build --output-on-failure`
  (at minimum `test-backend-ops` must still pass for MUL_MAT_ID on CPU).
- For end-to-end checks you need a small MoE GGUF. If network access to Hugging
  Face works, use the smallest available MoE model (e.g. a small Qwen MoE or
  OLMoE at Q4). If it does not, do not fake results: write unit-level tests and
  state clearly in the summary that end-to-end was not run.
- Never commit model files (`*.gguf`) or large trace files. Add them to
  `.gitignore` if needed. Commit only small sample traces (< 1 MB) for tests.
- Real measurements (CUDA, real models) are done by the owner on a Windows
  machine. Everything you write must build on Windows (MSVC) too: no POSIX-only
  APIs in C/C++ code, use `std::` facilities.

## Task 1 - Expert trace tool (do this first)

Create a new tool `tools/expert-trace/` (binary `llama-expert-trace`), modeled on
`examples/eval-callback`. Do not modify core library behavior.

- Use the eval callback (`cb_eval` / `cb_eval_user_data` in context params).
- In the "ask" phase, return true only for tensors whose name starts with
  `ffn_moe_topk-` (set in `build_moe_ffn` via `cb(selected_experts, "ffn_moe_topk", il)`).
  Verify the exact name in `src/llama-graph.cpp` before relying on it; if it
  differs, use the actual name and document it here.
  Verified: the name is `ffn_moe_topk-<il>` (`llama_context::graph_get_cb` formats `"%s-%d"`).
  It is a non-contiguous view of the argsort result, so index it with `nb[0]`/`nb[1]`.
- In the "data" phase, copy the tensor to host with `ggml_backend_tensor_get`.
  It is I32 with shape `[n_expert_used, n_tokens]`. Parse layer index from the
  name suffix.
- Also check how `tools/imatrix` collects per-expert statistics; reuse its
  approach where it fits instead of reinventing it.
- Output: CSV to a path given by `--trace-out <file>`:
  `ubatch,phase,layer,token,rank,expert` where `phase` is `prefill` or `decode`
  (decode = ubatch with 1 token). One row per (token, rank).
- Also write a small JSON sidecar with model metadata: model file name,
  `n_layer`, `n_expert`, `n_expert_used`, and per-layer bytes of one expert
  (gate+up+down, at the model's actual quant type). Get sizes from the loaded
  tensors (`ggml_nbytes(ffn_up_exps) / n_expert`, etc.).
- The tool otherwise behaves like `llama-cli`/`llama-simple`: takes `-m`, `-p` or
  `-f`, `-n`, `-ngl`.
- Add the target to the relevant `CMakeLists.txt`.

Done when: builds on CPU, running on a small MoE model produces a CSV whose row
count equals `n_layer_moe * n_tokens * n_expert_used`, and a sample trace is
committed under `tools/expert-trace/sample/`.

Status: done. See `tools/expert-trace/README.md`. The prompt is decoded with
output for all tokens (in `n_ubatch` chunks), otherwise the last layer only runs
the output tokens. Sample traces come from a tiny random-weight MoE (Hugging Face
is not reachable from the sandbox), so they test the tool, not real routing.

## Task 2 - Trace analysis (Python)

Create `tools/expert-trace/analyze.py` (Python 3, only stdlib + numpy +
matplotlib; no pandas requirement). Inputs: one or more CSV traces + JSON sidecar.

Report per layer and aggregate:

1. Selection histogram per expert.
2. Coverage: how many experts cover 50% / 80% / 95% of selections.
3. Cross-workload stability: for two traces A and B (e.g. Serbian text vs code),
   Jaccard overlap of their top-N hot sets per layer.
4. **Static placement simulation:** given a VRAM budget in bytes, pick the hottest
   experts per layer from profile A (greedy by count/bytes across layers) and
   report the hit rate on trace B. This is the key number for Phase 1.
5. **Dynamic cache simulation** (for deciding on Phase 2): per-layer LRU and LFU
   with K slots; report hit rate and bytes transferred per decoded token.
6. **Throughput bound for cold tier:** for a given tier bandwidth (MB/s: NVMe
   3000, SATA SSD 500, HDD 150 as presets), tokens/s upper bound =
   bandwidth / average bytes missed per token.

Output: a text summary to stdout and PNG plots to an output directory. Include
a `--help` and a short README section with example commands.

Done when: runs on the committed sample trace and produces all six sections.

Status: done. `analyze.py` runs on the sample traces and prints all six sections.

## Task 3 - Phase 1 design (do NOT implement until asked)

Hot/cold split per layer, driven by a placement file:

- Placement file (JSON): per layer, list of hot expert IDs and target device for
  hot and cold buckets. Generated by `analyze.py` from a profile.
- At load time, slice the merged `ffn_{gate,up,down}_exps` tensors into
  `exps_hot [.., n_hot]` (target device, e.g. CUDA) and `exps_cold [.., n_cold]`
  (CPU or mmap'd). Works with modern merged GGUFs; no special file format needed.
- Remap table per layer: global expert ID -> (bucket, local ID).
- In `build_moe_ffn`: split selected IDs into hot/cold, run MUL_MAT_ID on each
  bucket with remapped IDs, combine with routing weights. Exactly equivalent
  output to the unsplit path (verify numerically on CPU with both buckets on
  CPU: max abs diff must be ~0).
- Feature must be off by default; enabled only by `--moe-placement <file>`.
- RPC devices need no special handling: they are ggml backend devices and can be
  a bucket target like any other.

## Coding rules

- Minimal diffs in core files; prefer new files under `tools/`.
- Follow existing llama.cpp style (4 spaces, `snake_case`, no exceptions in
  hot paths).
- Every feature off by default; no behavior change for users who do not opt in.
- Small, focused commits with clear messages.
- At the end of each session, write a short summary: what was done, what was
  verified and how, what was NOT verified, open questions.
