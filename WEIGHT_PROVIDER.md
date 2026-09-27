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

Status: done. `analyze.py` runs on the sample traces and prints all sections (section 7, decode estimate, added after Phase 0 results).

## Phase 0 results (real model)

Measured by the owner: Qwen3-30B-A3B (48 MoE layers, 128 experts, top-8), RTX 2060 Super 8 GB, experts on CPU.

- Within one workload routing is concentrated: 80% of selections are covered by ~26-34 experts per layer (uniform would need ~103).
- Across workloads the hot sets are nearly disjoint: Jaccard of top-N sets is 0.04, lower than random sets (0.14).
- LRU with 32 slots per layer gives 82-89% hits.

What this means:

- A static hot set from one workload does not carry over to another. Phase 1 (static, profile-guided) only helps if the profile matches the workload, or is built from a mix of workloads (`analyze.py` accepts comma-separated trace groups for this).
- The hot set shifts between workloads, but inside one workload a small LRU cache holds most of it. This is the condition the plan gave for Phase 2 (dynamic residency).
- Next step before any design decision: section 7 of `analyze.py` (decode tokens/s estimate, LRU in VRAM vs all on CPU) on these traces, with the real PCIe / CPU / GPU numbers of the test machine.

## Task 3 - Phase 1: static hot/cold placement

Scope: `qwen3moe` only. Off by default, enabled by `--moe-placement <file.json>` (common arg, so it works in `llama-cli`, `llama-completion`, `llama-server`, `llama-batched-bench`, `llama-perplexity`, `llama-debug`, ...). `llama-bench` has its own argument parser and does not take it.

Status: implemented; verified on CPU and with an RPC device standing in for the GPU (see "Verification"). Not run on CUDA yet.

Pieces:

- `analyze.py --emit-placement <file>` writes the section 4 hot set (profile A, `--vram-budget`) as `{"layers": [{"layer": L, "hot": [ids]}, ...]}`. A layer that is missing from the file has no hot experts; `{"layers": []}` puts all experts in the cold bucket. There is no per-bucket device field: the hot bucket goes to the GPU if there is one, the cold bucket stays on the CPU.
- `llama_model_params::moe_placement / moe_hot / n_moe_hot` (C API): pairs (layer, expert) in the hot bucket.
- Loader (`src/llama-moe-placement.{h,cpp}`, hook in `llama_model_base::create_tensor_moe_placement`): the merged `ffn_{gate,up,down}_exps` tensors are not created. For each layer two tensors with the same row layout are created, `<name>.hot [.., n_hot]` and `<name>.cold [.., n_cold]` (an empty bucket gets no tensor), plus three I32 tables `[1, n_expert]`:
  - `ids_hot`: local id in the hot bucket, 0 for cold experts
  - `ids_cold`: local id in the cold bucket, -1 for hot experts
  - `bucket`: 0 = hot, 1 = cold
  The expert slices are copied from the file (mmap or read) before the normal tensor load, and the pages of the merged tensors are released after that.
  - Hot bucket buffer: the layer device if it is a GPU, else the first GPU, else the CPU (first buffer type that supports MUL_MAT_ID for it).
  - Cold bucket buffer: the CPU buffer types (CPU_REPACK is used when the type supports it). As in the normal loader, a host (pinned) buffer is replaced by the plain CPU buffer when mmap is used, so an all-cold placement uses the same buffers as `--cpu-moe`.
  - Id tables: on the hot bucket's device when the layer has hot experts (first buffer type there that supports GET_ROWS on I32), else on the CPU. See "Graph splits per layer".
  - `-ot` / `--cpu-moe` / `--n-cpu-moe` do not apply to the sliced tensors. Expert scale tensors (NVFP4 `.scale`) and LoRA on expert tensors are not supported with the placement.
- ggml: `ggml_mul_mat_id_set_skip(node, true)` (op param 4). With the flag, an id < 0 skips that slot and its dst row is set to 0. Implemented in the CPU backend (generic kernel and repack). `ggml_backend_dev_supports_op` / `ggml_backend_dev_offload_op` return false for flagged nodes on every non-CPU device; the spacemit extra buffer type also rejects them. Test: `test-backend-ops -o MUL_MAT_ID` cases with `skip=1`.
- Graph (`build_moe_ffn`, new `moe_pl` argument, passed only by qwen3moe): the expert FFN (up, gate, swiglu, down) runs once per bucket with local ids.

How the two buckets are combined (per layer, K = n_expert_used, T = n_tokens):

1. `ids_flat = cont(selected_experts)` as `[K*T]` (global ids).
2. Hot bucket: `ids_hot = get_rows(ids_hot_table, ids_flat)`. Cold slots get local id 0, so a token can hold the same hot id twice. CPU repack and CUDA MMQ assume unique experts per token (the repack row map overflows, CUDA `mm_ids_helper` records one slot), so when the layer also has cold experts the hot MUL_MAT_ID uses one slot per row: ids `[1, K*T]` and the input repeated to `[n_embd, 1, K*T]`. Without cold experts the normal `[K, T]` form is used. The hot bucket is not flagged, so its cold-slot rows hold real expert-0 output.
3. Cold bucket: `ids_cold = get_rows(ids_cold_table, ids_flat)` as `[K, T]`; hot slots are -1. The three cold MUL_MAT_IDs are flagged (only when the layer also has hot experts), so hot slots cost no CPU compute and their rows are 0.
4. Select, not multiply: `both = concat(hot, cold)` as `[n_embd, 2, K*T]` (row 0 = hot, row 1 = cold), `bucket = get_rows(bucket_table, ids_flat)` as `[1, K*T]`, `experts = get_rows(both, bucket)`. Each slot copies the row of the bucket that holds its expert; the other row is never read, so an inf/nan there cannot leak in (no `0*inf`).
5. `experts` is reshaped to `[n_embd, K, T]` and goes through the unchanged path: multiply by the router weights, sum over K.

All-hot and all-cold layers skip steps 4 and the repeat; all-cold layers also do not set the skip flag.

### Graph splits per layer

A split is a run of graph nodes on one backend; at each split boundary the scheduler copies the inputs over and waits for the previous backend. Measured with `sched_reserve: graph splits = N` (`-v`) on the tiny model (4 MoE layers) with an RPC device as the "GPU" and `-ngl 99`; the same count at `-ub 512` and `-ub 1`. The 2 splits of the no-override run are the input and output ends, the rest is per layer.

| run | total | per MoE layer |
|---|---|---|
| no override (all experts on the device) | 2 | 0 |
| `--cpu-moe` (= `-ot exps=CPU`, `--n-cpu-moe <all>`) | 10 | 2 |
| placement, all cold | 10 | 2 |
| placement, mixed, id tables on CPU (first version) | 18 | 4 |
| placement, mixed, id tables on the hot device (now) | 10 | 2 |
| placement, all hot | 2 | 0 |

Per layer, `--cpu-moe`: GPU [attention, router, top-k] -> CPU [up, gate, swiglu, down] -> GPU [weights, sum, next layer]. Copies: `ffn_norm` output `[n_embd, T]` F32 and the selected ids `[K, T]` I32 to the CPU, expert output `[n_embd, K, T]` F32 back.

Per layer, mixed placement (now): GPU [attention, router, top-k, the three id `get_rows`, hot FFN] -> CPU [cold FFN, flagged] -> GPU [concat, select, weights, sum, next layer]. Copies: `ffn_norm` output and `ids_cold` `[K, T]` I32 to the CPU, cold output `[n_embd, K, T]` F32 back. Same number of splits and the same copy sizes as `--cpu-moe`. With the tables on the CPU (first version) the id `get_rows` ran on the CPU before the hot FFN, which added a CPU -> GPU round trip per layer.

The remaining split per layer is the cold bucket itself and cannot go away. Two limits of the scheduler:

- Splits run one after the other: the hot FFN on the GPU and the cold FFN on the CPU do not overlap (the input copy to the CPU split waits for the GPU stream). Overlap needs scheduler changes.
- The counts above are for the RPC device, which does not offload ops. On CUDA, `--cpu-moe` at batch >= 32 offloads the CPU MUL_MAT_IDs to the GPU (weights copied over PCIe), so its prefill graph has other splits; flagged cold nodes are never offloaded (see "Expected costs").

### Verification done

CPU, tiny random qwen3moe (4 layers, 16 experts, top-4, n_embd 256, qwen2 vocab; F16, Q4_0, Q4_K_M; models not committed):

- Logits for a 40-token prompt (one batch) plus 8 greedy decode steps, placement vs unsplit, `n_ubatch` 512 and 16, placements: all hot, all cold, 5 random hot per layer, mixed layers (layer 0 all hot, layer 1 all cold, others half), 1 hot per layer, all but 1 hot per layer.
  - F16: max abs diff 0 for all.
  - Q4_0 / Q4_K_M with extra buffer types off: 0 for all.
  - Q4_0 / Q4_K_M with CPU_REPACK (default): 0, except when a bucket has exactly 1 expert: up to 3e-5 (logits up to ~190). A 1-expert tensor is 2D, so CPU_REPACK does not take it and the generic kernel runs instead of the repack kernel of the baseline.
- `llama-expert-trace` -> `analyze.py --emit-placement` (3 MiB budget, 26 of 64 experts hot) -> `llama-debug --moe-placement`: logits equal to the unsplit path (diff 0), with mmap and with `--load-mode none`.
- `test-backend-ops -b CPU -o MUL_MAT_ID`: all pass, including the `skip=1` cases.
- CPU-only speed, tiny Q4_K_M model, `llama-batched-bench -npp 512 -ntg 128`: all cold = baseline (9460 vs 9640 t/s pp, 354 vs 350 t/s tg); mixed (26/64 hot, both buckets on the CPU): pp 5930 t/s (-38%), tg 332 t/s (-5%), because without a GPU the hot bucket also runs on the CPU and computes all K*T slots. Not the target setup, only shows the overhead.
- Before step 2 used one slot per row, mixed placements with CPU_REPACK were wrong (diff ~70), from the duplicate ids above.

RPC device as the "GPU" (`ggml-rpc-server` with the CPU backend on the same machine, `--rpc 127.0.0.1:50052 -ngl 99`), with `compare-logits.py` (87-token prompt, `-ub 512` and `-ub 1`):

- The graph split counts above.
- Flagged cold nodes stay on the CPU (the split counts show it; a flagged node on the RPC device would have failed the exact checks below).
- All cold vs `--cpu-moe`: exact (0) at both ubatch sizes.
- With the same kernels everywhere (`-nr`, so neither side uses CPU_REPACK): mixed placement (hot bucket and id tables on the RPC device) and all hot vs `--cpu-moe`: exact (0) at both ubatch sizes, also when the prompt is split over several ubatches.
- With CPU_REPACK on the client (default): mixed placement vs `--cpu-moe` differs by 1.5e-5 at `-ub 512` but by 1.17 at `-ub 1`. Every RPC run, also without placement (`--cpu-moe`, `--n-cpu-moe 2`, no override), differs from the CPU-only run by ~1.5 at `-ub 16`, and all of them match exactly with `-nr`. So this is kernel noise (plain kernels on the RPC server vs repack on the client), made large because on a random-weight model a 1e-5 change flips near-tied router choices. This is why the CUDA check below uses exact checks plus KL-divergence over many tokens, not a single-prompt tolerance.

Not verified:

- Anything on CUDA: hot bucket and id tables on CUDA, CUDA rejecting flagged nodes, the one-slot-per-row hot MUL_MAT_ID on CUDA (MMQ/MMVQ with `n_expert_used = 1`), CUDA GET_ROWS on the I32 tables.
- A real model, KL-divergence numbers, and any speed numbers.
- MSVC build.

### Correctness check tools

`tools/expert-trace/compare-logits.py` runs `llama-debug --save-logits` once per named run, at each `--ubatch` size (default `512,1`: one batch like prefill, and one token per graph like decode), and compares the last-token logits with the first run.

- `--exact NAME`: that run must match the reference bit for bit. Use it only for runs that compute every op with the same kernels.
- Other runs are only reported (max abs diff, top-1, top-10 overlap), or checked against `--tol X` if given.

Expected tolerance:

- Exact (0): all-cold placement vs `--cpu-moe`, at every ubatch size, on CPU and on CUDA. A non-zero diff here is a bug.
- CPU-only (no GPU, `-nr` or F16): any placement vs unsplit is exact. With CPU_REPACK, 1-expert buckets differ by ~1e-5 relative.
- Hot experts on CUDA: not exact (CUDA kernels quantize activations to q8_1 and sum in another order). Judge by KL-divergence against a run that moves experts to the GPU without placement (`--n-cpu-moe 40` vs `--cpu-moe`): the placement's mean KLD should be of the same order (within ~2x) and `Same top p` within about a point of that run. Signs of a bug: mean KLD above ~0.01, or a PPL ratio off by more than ~1%. The noise level on Qwen3-30B is not measured yet; record it in step 3.

### Measurement sequence (owner machine: RTX 2060 Super 8 GB, 32 GB RAM, Qwen3-30B-A3B Q4_K_M)

Paths are written POSIX style; on Windows the binaries are in `build\bin\Release\` and Python is `python`. `M` is the model file, the same file for trace, placement and runs (expert ids and bytes come from it). Keep all runs at the same `-c` and threads (`-t` = physical cores).

1. Build: `cmake -B build -DGGML_CUDA=ON` and `cmake --build build --config Release -j`.

2. Traces (skip if the Phase 0 traces of this model file are still there). Use different text for the traces than for the speed runs in step 5:
   ```
   llama-expert-trace -m M -ngl 99 --cpu-moe -c 4096 -f prompt_sr.txt   -n 256 --trace-out trace_sr.csv
   llama-expert-trace -m M -ngl 99 --cpu-moe -c 4096 -f prompt_code.txt -n 256 --trace-out trace_code.csv
   llama-expert-trace -m M -ngl 99 --cpu-moe -c 4096 -f prompt_chat.txt -n 256 --trace-out trace_chat.csv
   ```

3. Placements. Profile A = the mix of the three traces, B = one of them (B only affects the printed hit rates, not the file). `--vram-budget` counts expert bytes only:
   ```
   python tools/expert-trace/analyze.py trace_sr.csv,trace_code.csv,trace_chat.csv trace_code.csv --no-plots --vram-budget 3G --emit-placement placement-3g.json
   python tools/expert-trace/analyze.py trace_sr.csv,trace_code.csv,trace_chat.csv trace_code.csv --no-plots --vram-budget 5G --emit-placement placement-5g.json
   echo '{"layers": []}' > all-cold.json
   ```
   (Windows cmd: `echo {"layers": []} > all-cold.json`, without the single quotes.)
   Note the section 4 hit rates. For reference: `--n-cpu-moe 40` keeps the experts of 8 whole layers on the GPU, which is close to 3 GB of experts; 5 GB is about 13 layers (`--n-cpu-moe 35` would be the same-VRAM baseline for it).

4. Correctness on CUDA, before any speed run.
   a. Exact check and a report of the placements (`-ub 512` and `-ub 1`):
   ```
   python tools/expert-trace/compare-logits.py --llama-debug build/bin/llama-debug -m M \
       --run "ref=--cpu-moe" --run "cold=--moe-placement all-cold.json" \
       --run "noise=--n-cpu-moe 40" --run "p3=--moe-placement placement-3g.json" --run "p5=--moe-placement placement-5g.json" \
       --exact cold -- -ngl 99 -c 4096
   ```
   Expected: `cold` exact (result OK); `noise`, `p3`, `p5` reported with the same top-1 and small diffs. A single prompt can show one larger diff from a flipped router choice, so the decision is made in b.
   b. KL-divergence on wikitext-2 (`scripts/get-wikitext-2.sh`, or download `wikitext-2-raw-v1.zip` from `huggingface.co/datasets/ggml-org/ci` and use `wiki.test.raw`):
   ```
   llama-perplexity -m M -ngl 99 -f wiki.test.raw -c 512 --chunks 40 --cpu-moe --kl-divergence-base ref.kld
   llama-perplexity -m M -ngl 99 -f wiki.test.raw -c 512 --chunks 40 --n-cpu-moe 40                      --kl-divergence-base ref.kld --kl-divergence
   llama-perplexity -m M -ngl 99 -f wiki.test.raw -c 512 --chunks 40 --moe-placement placement-3g.json --kl-divergence-base ref.kld --kl-divergence
   llama-perplexity -m M -ngl 99 -f wiki.test.raw -c 512 --chunks 40 --moe-placement placement-5g.json --kl-divergence-base ref.kld --kl-divergence
   ```
   Record `Mean KLD`, `99.9% KLD`, `Same top p` and `Mean PPL(Q)/PPL(base)` of each. Pass: the placements are within the tolerance above, relative to the `--n-cpu-moe 40` run. This covers the prefill path (batches of 512); a. covers `-ub 1`.
   Also check the load log: `moe placement buffer size` lines show the hot bytes in `CUDA0` and the cold bytes in `CPU_REPACK`/`CPU`. If 5G does not fit in VRAM with `-c 4096`, use 4.5G.

5. Speed. `llama-batched-bench` feeds random tokens, so its routing matches no workload and it shows raw throughput; the placement only pays off when routing matches the profile, so also measure on real text. Run each line 3 times and take the median; watch that nothing else uses the GPU.
   ```
   llama-batched-bench -m M -ngl 99 -c 4096 -npp 512 -ntg 128 -npl 1 --n-cpu-moe 48
   llama-batched-bench -m M -ngl 99 -c 4096 -npp 512 -ntg 128 -npl 1 --n-cpu-moe 40
   llama-batched-bench -m M -ngl 99 -c 4096 -npp 512 -ntg 128 -npl 1 --moe-placement placement-3g.json
   llama-batched-bench -m M -ngl 99 -c 4096 -npp 512 -ntg 128 -npl 1 --moe-placement placement-5g.json
   ```
   Record `S_PP t/s` (pp512) and `S_TG t/s` (tg128). Real text, same four configurations, with a held-out prompt of the profiled workloads (and one of another workload to see the effect of a profile mismatch):
   ```
   llama-completion -m M -ngl 99 -c 4096 -f heldout_code.txt -n 128 --temp 0 --ignore-eos -no-cnv --n-cpu-moe 48
   ... same with --n-cpu-moe 40, --moe-placement placement-3g.json, --moe-placement placement-5g.json
   ```
   Record `prompt eval time` (t/s) and `eval time` (t/s) from the perf lines at the end.

6. Write the numbers into this file: VRAM used (model + placement + compute buffers from the load log), pp and tg for each configuration, the hit rates from step 3, and the KLD results from step 4.

### Expected costs to check in the numbers

- Prefill regression (pp512): `--n-cpu-moe` / `--cpu-moe` offload the CPU MUL_MAT_IDs to the GPU at batch >= 32 (weights copied over PCIe). Flagged cold nodes are rejected by the GPU, so they always run on the CPU. pp512 with the placement can be slower than `--n-cpu-moe 48`. Measure pp512 as well as tg128.
- Hot bucket waste: cold slots are computed on the GPU with hot expert 0 and dropped, and with both buckets the hot input is repeated K times.
- No overlap: the hot FFN (GPU) and the cold FFN (CPU) of a layer run one after the other.

Open questions:

- If pp512 regresses a lot: allow offload of flagged nodes to a backend that implements the skip (CUDA support for the flag), or drop the flag for large batches.
- Overlap of the GPU and CPU buckets needs changes in the scheduler.
- Whether the repack row map should be fixed for duplicate ids per token instead of the one-slot-per-row workaround (upstream code, not changed here).

## Coding rules

- Minimal diffs in core files; prefer new files under `tools/`.
- Follow existing llama.cpp style (4 spaces, `snake_case`, no exceptions in
  hot paths).
- Every feature off by default; no behavior change for users who do not opt in.
- Small, focused commits with clear messages.
- At the end of each session, write a short summary: what was done, what was
  verified and how, what was NOT verified, open questions.
