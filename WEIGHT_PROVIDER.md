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
  The hot expert slices are copied from the file (mmap or read) before the normal tensor load.
  - Hot bucket buffer: the layer device if it is a GPU, else the first GPU, else the CPU (first buffer type that supports MUL_MAT_ID for it).
  - Cold bucket (current state, see Task 7): with mmap it is the merged `ffn_*_exps` tensor itself, left in the file mapping (plain CPU buffer, like the experts of `--cpu-moe` / `--n-cpu-moe`), used with global ids and the experts of the other buckets skipped; nothing is copied. Without mmap (`--load-mode none`) the cold experts are copied into a host buffer (pinned host buffer if there is a GPU, else plain CPU). The first version copied them into CPU_REPACK.
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

Unattended: `powershell -ExecutionPolicy Bypass -File tools\expert-trace\measure-windows.ps1` runs the steps below with the paths of the owner machine (model `D:\AInode2\models\Qwen3-30B-A3B-Q4_K_M.gguf`, binaries `build\bin\Release`, traces `tr_sr_pravni.csv`, `trace_sr.csv`, `trace_code.csv` with their `.json` sidecars in the repo root; all are parameters):

- aborts if free RAM is under 20 GB (`-MinFreeRamGB`), after printing free RAM and page file use;
- placements 3G / 5G / ranking, correctness (exact all-cold vs `--cpu-moe` plus a report of the others, then KLD on wikitext-2, downloaded if missing), `llama-batched-bench` for `--n-cpu-moe 48`, `--n-cpu-moe 40`, the 3G and 5G hot sets and the ranking with `--moe-vram-margin 1G` at `-c 4096` (`-Repeats`, default 3), real-text decode with `llama-completion` (code prompt from `src/llama-sampler.cpp`; text prompt `heldout_sr.txt` in the repo root if present, else a part of README.md), one `-v` run per configuration for the buffer sizes, and `llama-bench -ncmoe 48,44,40,36` with `-fa off,on` for f16 KV and `-fa on` for q8_0 KV (a quantized V cache needs flash attention);
- every step logs to `results\moe-<date>\NN-<step>.log` with start and end time, `steps.log` has the timeline, a failed step is recorded and the script goes on; `summary.md` has the step table, compare-logits result, KLD table, batched-bench medians, decode speeds, placement reports with buffer sizes and the llama-bench tables.

Tested with PowerShell 7.4 on Linux against the Linux binaries and the tiny model (`Get-CimInstance` mocked), including the RAM abort; not run on Windows PowerShell 5.1 or on the real model. The script is written for 5.1 (no PowerShell 7 syntax).

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
   python tools/expert-trace/analyze.py trace_sr.csv,trace_code.csv,trace_chat.csv trace_code.csv --no-plots --emit-ranking ranking.json
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
       --run "rank=--moe-placement ranking.json" --exact cold -- -ngl 99 -c 4096
   ```
   Expected: `cold` exact (result OK); `noise`, `p3`, `p5` reported with the same top-1 and small diffs. A single prompt can show one larger diff from a flipped router choice, so the decision is made in b.
   b. KL-divergence on wikitext-2 (`scripts/get-wikitext-2.sh`, or download `wikitext-2-raw-v1.zip` from `huggingface.co/datasets/ggml-org/ci` and use `wiki.test.raw`):
   ```
   llama-perplexity -m M -ngl 99 -f wiki.test.raw -c 512 --chunks 40 --cpu-moe --kl-divergence-base ref.kld
   llama-perplexity -m M -ngl 99 -f wiki.test.raw -c 512 --chunks 40 --n-cpu-moe 40                      --kl-divergence-base ref.kld --kl-divergence
   llama-perplexity -m M -ngl 99 -f wiki.test.raw -c 512 --chunks 40 --moe-placement placement-3g.json --kl-divergence-base ref.kld --kl-divergence
   llama-perplexity -m M -ngl 99 -f wiki.test.raw -c 512 --chunks 40 --moe-placement placement-5g.json --kl-divergence-base ref.kld --kl-divergence
   llama-perplexity -m M -ngl 99 -f wiki.test.raw -c 512 --chunks 40 --moe-placement ranking.json      --kl-divergence-base ref.kld --kl-divergence
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

   Load-time budget (Task 4), same two tools, with `-c 4096` (the budget subtracts the KV cache of the actual context):
   ```
   llama-batched-bench -m M -ngl 99 -c 4096 -npp 512 -ntg 128 -npl 1 --moe-placement ranking.json
   llama-batched-bench -m M -ngl 99 -c 4096 -npp 512 -ntg 128 -npl 1 --moe-placement ranking.json --moe-ram-pin auto
   ```
   Copy the `moe placement from ranking` report (budget line and tier table) from the log. Check that the run did not fail with out-of-memory; if it did, raise `--moe-vram-margin` (e.g. 1G) and note the value. The `--moe-ram-pin` run uses the file mapping for the cold bucket (no CPU_REPACK), so compare it with the ranking run without it: the difference is the cost of the plain CPU kernels. With `-v`, the `lock: moe placement: RAM tier ...` line says how much was locked; on Windows a failed `VirtualLock` is printed as a warning.

6. Write the numbers into this file: VRAM used (model + placement + compute buffers from the load log), pp and tg for each configuration, the hit rates from step 3, the KLD results from step 4, and the ranking report (budget and tiers).

### Expected costs to check in the numbers

- Prefill regression (pp512): `--n-cpu-moe` / `--cpu-moe` offload the CPU MUL_MAT_IDs to the GPU at batch >= 32 (weights copied over PCIe). Flagged cold nodes are rejected by the GPU, so they always run on the CPU. pp512 with the placement can be slower than `--n-cpu-moe 48`. Measure pp512 as well as tg128.
- Hot bucket waste: cold slots are computed on the GPU with hot expert 0 and dropped, and with both buckets the hot input is repeated K times.
- No overlap: the hot FFN (GPU) and the cold FFN (CPU) of a layer run one after the other.

Open questions:

- If pp512 regresses a lot: allow offload of flagged nodes to a backend that implements the skip (CUDA support for the flag), or drop the flag for large batches.
- Overlap of the GPU and CPU buckets needs changes in the scheduler.
- Whether the repack row map should be fixed for duplicate ids per token instead of the one-slot-per-row workaround (upstream code, not changed here).

## Task 4 - load-time VRAM budget and RAM tier

Scope: as Task 3 (qwen3moe, off by default). All of it is used only with `--moe-placement <ranking.json>`.

### Reuse check (done before writing code)

- `common_fit_params` (`--fit`, on by default) changes `-ngl`, the tensor split, per-layer expert overrides (`-ot` style) and `n_ctx` when they are left at their defaults. It does not pick single experts, so it cannot make the hot set. The ranking budget runs after it, on the final `mparams` / `cparams`, so the fit decides the layer split and the context first.
- `common_get_device_memory_data` (common/fit.h) loads the model and a context with `no_alloc` and returns per device the free memory and the model, context (KV) and compute buffer bytes; the compute buffers are measured by the graph reserve of that context. This is exactly what the budget needs, so it is reused as is; no new estimator.
- The CPU device reports free = total on Linux ("ill-defined"), so the RAM budget reads `MemAvailable` from `/proc/meminfo`; on Windows it uses min(available physical, available commit).
- colibri (github.com/JustVugg/colibri, `c/resource_plan.py`), read for ideas only, no code taken: it sizes the tiers with fixed overhead constants (runtime 1.2 + 2.5 GB, 2 GB GPU reserve, RAM budget 88 % of available) and prices experts per layer uniformly, without locking. Taken over as ideas: on Windows the commit limit matters as much as free physical memory (colibri hit it), and a fixed reserve for the rest of the system. Different here: overheads are measured by a dry run instead of constants, experts are chosen one by one by count per byte, and the RAM tier is locked.

### What it does

1. `analyze.py --emit-ranking ranking.json`: every expert of every layer with its selection count in profile A and its bytes.
2. `--moe-placement ranking.json` (the file format is detected: layers with `"experts"` are a ranking, layers with `"hot"` a fixed hot set). In `common_init_result`, after `--fit`:
   - Expert bytes are read from the model file (GGUF tensor sizes); a warning is printed if the ranking has other sizes (other model or quant).
   - Dry run: `common_get_device_memory_data` with a probe placement (the top expert of every layer hot), so every layer has both buckets and the compute buffers are measured for the mixed graph, the largest case.
   - VRAM budget on the first GPU: `free - (model - probe experts) - KV - compute - --moe-vram-margin` (default 512M). With more than one GPU only the first is budgeted (warning).
   - Greedy fill over all layers by count per byte (unseen experts last, spread over layers, as analyze.py section 4); an expert that does not fit is skipped and smaller ones can still fit.
3. `--moe-ram-pin <size|auto>`: the next experts in the same order go to the RAM tier up to the size. `auto` = available RAM - host memory of the rest of the model (model without the cold experts, host KV, host compute, from the dry run) - 2 GiB reserve. In the loader:
   - With mmap the cold bucket is the merged `ffn_*_exps` tensor in the file mapping (since Task 7 also without `--moe-ram-pin`). The RAM tier experts are locked in it with `llama_mlock` (mlock / VirtualLock, 64 KiB aligned runs of experts); the rest is not locked and the OS pages it in and out (disk tier).
   - Without mmap (`--load-mode none`) the cold experts are copied and the RAM tier is locked in that copy; there is no disk tier.
   - Result: `lock: moe placement: RAM tier N experts, X MiB requested, Y MiB locked in R ranges` (info, shown with `-v`), or a warning with how much was locked when a lock call failed (always shown). After the first failure no more locks are tried.
4. Report, printed by common before the load:
   ```
   moe placement from ranking (profile tq.csv, 704 selections, 64 experts)
     VRAM budget: RPC0: free 16095 MiB - trunk 30 MiB - KV 2 MiB (n_ctx 1024) - compute 297 MiB - margin 15763 MiB = 2 MiB
     RAM budget: 2 MiB (set by --moe-ram-pin)
     tier          experts          MiB    share of selections
     VRAM (hot)         21          2.4                  62.2%
     RAM (pinned)       18          2.0                  22.7%
     disk (mmap)        25          2.9                  15.1%
   ```
   Without `--moe-ram-pin` the last row is `RAM (cold)`. The share is the expected hit rate if the workload routes like the profile. The loader adds (with `-v`) `load: moe placement: hot N experts (X MiB), cold M experts (Y MiB, in the file mapping | copied, no mmap)`.

Notes:

- Set `-c`. With `n_ctx = 0` the context is the training context (40960 for Qwen3-30B-A3B), whose KV cache takes VRAM from the experts; a warning is printed. `--fit` may also lower the context before the budget is computed.
- `-ot`, `--cpu-moe`, `--n-cpu-moe` do not apply to the placement tensors; do not combine them with `--moe-placement`.
- Linux: `RLIMIT_MEMLOCK` (`ulimit -l`, often 8 MB for users) limits locking unless the process has `CAP_IPC_LOCK`. Windows: `VirtualLock` needs a larger working set, which `llama_mlock` requests with `SetProcessWorkingSetSize`; it can still fail for large sizes.
- New API: `llama_model_params::moe_ram_pin`, `moe_warm`, `n_moe_warm`; `llama_mlock::failed()`. New files `common/moe-placement.{h,cpp}`.

### Verification done (CPU, and RPC device as the "GPU"; tiny Q4_K_M model, 4 layers x 16 experts)

- `--emit-ranking` on the sample-style trace: 4 layers x 16 experts, counts sum to the 704 selections of the trace.
- CPU only: report says no GPU, all 64 experts cold. `--moe-ram-pin auto` (available 14748 MiB) locks all 7.27 MiB; `--moe-ram-pin 3M` puts 26 experts (71 % of selections) in RAM and 38 on disk, 2.94 MiB locked in 45 ranges.
- Lock failure: with `CAP_IPC_LOCK` dropped (`setpriv`) and `ulimit -l 256`: `failed to mlock` warning and `only 0.18 MiB locked - locking FAILED`. As root with the capability, locking works regardless of `ulimit -l`.
- RPC device (reports 16095 MiB free): the budget line matches the real allocation of that run (trunk 30.94 MiB, compute 297.25 MiB measured by the dry run and by the real context alike). With the default margin all 64 experts are hot. With the margin set so that the budget is ~2.6 MiB: 21 experts hot (2.39 MiB, 62 % of selections), 18 pinned (2 MiB locked), 25 left in the mapping; graph splits 10, the same as `--cpu-moe`.
- Logits (`compare-logits.py`, `-ub 512` and `-ub 1`, same kernels everywhere with `-nr`), all exact (0) against the unsplit model: CPU only - ranking, ranking + `--moe-ram-pin 3M` / `auto` (mapped cold bucket), ranking + pin with `--load-mode none` (copied), the old hot-set file; RPC - ranking with partial budget, the same plus `--moe-ram-pin 2M` (three tiers), ranking with everything hot.

Not verified (needs the GPU machine):

- The budget on CUDA: that `free` from the CUDA device, the trunk, KV and compute from the dry run leave enough room, and that 512M is a good default margin (CUDA has allocations outside ggml buffers, e.g. cuBLAS workspace and the CUDA context).
- Locking large sizes (GBs) and `VirtualLock` on Windows; the effect of the disk tier when the model does not fit in RAM (Qwen3-30B-A3B Q4_K_M fits in 32 GB, so the disk tier stays empty with `auto`).
- Speed of the mapped cold bucket (plain kernels) against the copied one (CPU_REPACK).
- MSVC build of `common/moe-placement.cpp` (uses `GlobalMemoryStatusEx`).

## Task 5 - several hot devices (--moe-devices) [EXPERIMENTAL, not tested on real multi-GPU]

Opt-in; without `--moe-devices` nothing changes (one hot bucket per layer, on the layer device if it is a GPU, else on the first GPU).

- `--moe-devices <dev1,dev2,..>` (names as in `--list-devices`, e.g. `CUDA0,CUDA1` or `CUDA0,RPC0`), only with a ranking file. The devices must be used by the model (`-dev`, `--rpc`).
- Budget per listed device, the same way as Task 4, from that device's share of the dry run: `free - (model share - probe experts) - KV share - compute - margin`. The probe puts expert number s (by count) of every layer on device s, so every device's compute buffer is measured with its bucket in the graph.
- `--moe-vram-margin` takes one value for all devices or one per listed device (`1G,512M`).
- Fill: devices in the listed order, each greedily with the hottest unassigned experts (count per byte); the rest goes to the CPU cold bucket (and the RAM tier with `--moe-ram-pin`) as before.
- API: `llama_model_params::moe_devices` (NULL-terminated) and `moe_hot_dev` (device index of each hot pair).
- Report: one budget line and one tier row per device (`VRAM CUDA0`, `VRAM CUDA1`, ...); the loader adds (with `-v`) one line per device with experts and MiB.

Graph: a layer has one hot bucket per device that holds some of its experts, plus the cold bucket. `llama_layer_moe_placement` is now a list of buckets and one bucket table:

1. For each bucket b: `ids_b = get_rows(ids_table_b, ids_flat)`. Hot buckets (not flagged) map the other experts to local 0 and use one slot per row (ids `[1, K*T]`, input repeated), as in Task 3; the cold bucket is flagged and uses `[K, T]` with -1 for the other experts.
2. Each bucket output `[n_embd, K, T]` is reshaped to `[n_embd, 1, K*T]` and concatenated: `stacked [n_embd, N, K*T]`.
3. `bucket = get_rows(bucket_table, ids_flat)` (0..N-1), `experts = get_rows(stacked, bucket)`, then the unchanged weighting and sum. A layer with one bucket skips 3.

Placement of the id tables: each hot bucket's table on its device; the cold id table and the bucket table on the layer device when it is a GPU (else the first hot device). The hot bucket on the layer device comes first in the graph. With one device this is the same as before.

Graph splits, two RPC servers as stand-in GPUs (tiny model, 4 MoE layers; layers 0-2 on RPC0, layer 3 and the output on RPC1; same counts at `-ub 512` and `-ub 1`):

| run | splits |
|---|---|
| no override (all on the devices) | 3 |
| `--cpu-moe` | 10 |
| ranking, one hot device (default), everything fits | 3 |
| `--moe-devices RPC0,RPC1`, everything fits on RPC0 | 5 |
| `--moe-devices RPC1,RPC0`, both partial + cold, first version | 23 |
| same, tables on the layer device and its bucket first (now) | 19 |

Per layer with three buckets (layer on RPC0): RPC0 [attention, router, id lookups, RPC0 bucket] -> RPC1 [RPC1 bucket] -> CPU [cold bucket] -> RPC0 [combine, next layer], about 4 splits per layer against 2 for `--cpu-moe`. Every extra hot device of a layer adds one round trip, and the splits run one after the other (no overlap between devices).

Verification (two local `ggml-rpc-server`s, `-nr` so all runs use the same kernels):

- Logits exact (0) against `--cpu-moe`, at `-ub 512` and `-ub 1` (margins set per ubatch size so that both devices are partial): RPC1 + RPC0 + cold (e.g. 6 / 10 / 48 experts at `-ub 512`, 16 / 23 / 25 at `-ub 1`), the same with `--moe-ram-pin 1M`, everything on one listed device.
- 48-layer model (from `regress-placement.py`), three buckets per layer (355 / 340 / 73 experts): exact; graph 3822 nodes of 11460 allowed.
- Without `--moe-devices`: `regress-placement.py` on CPU and on one RPC device OK; single-RPC split counts unchanged (10 mixed, 2 all hot); the older CPU logits checks unchanged.
- The budget depends on the ubatch size (compute buffers), so a margin tuned at one `-ub` gives another fill at another `-ub`. That is expected.

Not verified: real multi-GPU (CUDA0 + CUDA1, or CUDA + a remote RPC GPU), speed, PCIe/network cost of the extra round trips.

## Graph size fix (crash on 48 layers)

A hot-set placement crashed on Qwen3-30B-A3B (48 MoE layers, CUDA) with `GGML_ASSERT(obj_new)`. Cause: the graph size limit is `8 x n_tensors`; with the placement the merged expert tensors were not counted any more, the placement tensors were not counted either, and each layer has more graph nodes. On the 4-layer test model the minimum of 1024 nodes hid it; on a 48-layer tiny model it reproduces as a hash set abort at graph reserve (the assert on the real model is the graph metadata buffer, sized from the same limit, while building the graph - not the loader's contexts, whose size does not change with the placement). Fix (commit "count placement tensors and nodes in the graph size"): the placement tensors are added to the model's tensor list, and `llama_model::n_moe_placement_nodes()` adds headroom per layer and bucket to `graph_max_nodes`. Regression check: `tools/expert-trace/regress-placement.py` (48 layers, hot-set and ranking, exact logits; `-- --rpc host:port -ngl 99` for a device); it fails without the fix and passes with it.

## Task 6 - follow-up to the first measurements (prefill, decode profile, KLD)

Owner results (Qwen3-30B-A3B Q4_K_M, RTX 2060 Super 8 GB): prefill with placement 73-110 t/s against 157-213 t/s for `--n-cpu-moe` (ranking: 21 t/s); decode with the ranking (70.8 % of the profile's selections in VRAM) only +18 % over `--n-cpu-moe 40` on in-domain text; KLD 0.009 for the placements against ~0 for `--n-cpu-moe 40` (both against `--cpu-moe`).

### 1. Prefill: two causes, both fixed

- The skip flag keeps the cold MUL_MAT_ID on the CPU (the GPU rejects flagged nodes), while `--n-cpu-moe` / `--cpu-moe` let the scheduler offload the CPU experts to the GPU at batch >= 32 (CUDA `GGML_OP_OFFLOAD_MIN_BATCH`).
- Found while fixing it: the copied cold bucket was in `CPU_REPACK`, and the scheduler only offloads ops whose weights are in a host buffer (`ggml_backend_buffer_is_host`; `CPU_REPACK` is not, its layout is not usable by a GPU). So even unflagged it could not be offloaded. The `--n-cpu-moe` experts are in the plain CPU buffer (mmap) or the pinned host buffer (no mmap).

Fix:

- `build_moe_ffn`: when `n_tokens >= GGML_OP_OFFLOAD_MIN_BATCH` (default 32, the CUDA default and the same variable) and op offload is on (not `--no-op-offload`), the cold bucket runs unflagged with a second id table (`ids_cold0`: 0 instead of -1 for the experts of the other buckets) and in the one-slot-per-row form (those zeros can repeat within a token). Below that it stays flagged, as before. No weight multiply is needed: the combine takes each slot's row by bucket id, so the dummy rows are never read.
- The cold bucket is allocated only in host buffer types, like the `--n-cpu-moe` experts: plain CPU buffer with mmap, pinned host buffer without mmap (since Task 7, with mmap it is the file mapping itself). When the scheduler offloads it, it copies only the experts used in the batch (existing logic), i.e. only cold experts, fewer bytes than `--n-cpu-moe`.
- Cost: at decode the cold experts use the plain CPU kernels instead of repack. `llama-cold-ffn-bench` here: repack is 10-20 % faster per expert at 4 threads (474 vs 574 us for 8 experts per layer).
- Verified here (no GPU, so the offload itself is not): switch at the threshold (a 30-token batch stays flagged, 512 unflagged, `--no-op-offload` always flagged, checked with a debug print), logits exact against the unsplit model at `-ub 512`, `16`, `1` on CPU and with the hot bucket on an RPC device, `regress-placement.py` OK. On a CPU-only machine the exact all-cold check needs `-nr` on both sides (there `--cpu-moe` uses `CPU_REPACK`); on the CUDA machine both use the plain CPU buffer.
- The ranking's 21 t/s is lower than the hot sets' 73-110 t/s; not explained by the flag alone. To check on the machine: VRAM use against the budget line (Task Manager, "shared GPU memory": the Windows driver can spill to system RAM when VRAM is oversubscribed, which is very slow) and the new pp numbers.

### 2. Decode: where the time goes (analysis; nothing changed for decode)

Checked in the code:

- CUDA graphs: not disabled by the placement on this GPU. The hot MUL_MAT_ID in the one-slot-per-row form has `ne[2] = K*T = 8` at decode; on Turing and newer Q4_K / Q6_K allow MMVQ with ids up to 8 (`get_mmvq_mmid_max_batch_turing_plus`), so no stream sync and CUDA graphs stay on (older GPUs or other quant types could fall back and disable them). The CUDA backend keeps one CUDA graph per split (keyed by the first node), so the per-layer splits do not force re-capture.
- Splits per token: the placement has a CPU split in every layer that has cold experts (48), `--n-cpu-moe 40` in 40 layers.
- CPU threadpool / barriers: `llama-cold-ffn-bench` (one layer's cold FFN, Qwen3-30B-A3B shapes, Q4_K) here: 20-28 us per layer with 0 active experts (wake-up, barriers, skip zeroing), then 60-75 us per active expert at 4 threads (37-44 GB/s). 48 layers x ~25 us = ~1.2 ms per token: not the main cost.
- Expected CPU expert work per token: `--n-cpu-moe 40` 40 x 8 = 320 expert FFNs; the ranking at 70.8 % 48 x 2.3 = ~112, about a third. If the CPU expert time were the bottleneck, decode would gain much more than +18 %.

So the missing time is elsewhere. Candidates, to be measured on the machine (the `-SkipProfile` section of `measure-windows.ps1` does it):

1. The real hit rate on the decode text is lower than the profile share (70.8 % is the share of the profile's selections, not of the test text). Check: trace the decode prompt with `llama-expert-trace` and compare with the hot set.
2. Per-split sync and copy latency (Windows WDDM), 48 CPU round trips per token: `GGML_SCHED_TIMING=2` shows copy+sync per split and layer.
3. Hot bucket GPU time (one-slot-per-row repeat, cold slots computed with hot expert 0 and dropped): GPU compute per layer in the same output.
4. Thread count: decode at `-t 4/6/8`.

New measurement tools:

- `GGML_SCHED_TIMING=1|2` (ggml scheduler, off by default): after each split the split backend is synchronized and its input copy (with waits) and compute are timed; 1 prints one line per graph with totals per backend, 2 also one line per split with the first named node (carries the layer number). At decode every GPU split is followed by a CPU split that waits for it anyway, so the extra syncs change little there.
- `tools/expert-trace/sched-timing-summary.py`: average per token by backend and a per-layer table from that output.
- `llama-cold-ffn-bench [repack] [threads ...]`: the CPU cold FFN microbenchmark above.
- The RPC stand-in is not usable for these timings: TCP instead of PCIe, and the rpc-server shares the CPU cores with the client.

### 3. KLD 0.009: likely causes and the test

- The proposed test (`--cpu-moe --load-mode none` vs `--cpu-moe` with mmap) does not isolate repack on the CUDA machine: without mmap the `--cpu-moe` experts go to the pinned host buffer (plain kernels, not `CPU_REPACK`), so both runs use the same kernels.
- A second, probably larger source: `llama-perplexity -c 512` runs batches of 512, where `--cpu-moe` and `--n-cpu-moe 40` offload all experts to the GPU (identical CUDA kernels, KLD ~0 between them), while the flagged cold bucket of the placements ran on the CPU (repack, q8_K activations) - a CPU-vs-GPU kernel difference for ~30 % of the expert work. The prefill fix removes both: the cold bucket is in a plain host buffer and is offloaded at batch 512 like the baseline, so the KLD of the placements should drop toward the `--n-cpu-moe 40` level.
- Decomposition added to `measure-windows.ps1` (at `-ub 16` nothing is offloaded): `kld-cpu-vs-gpu-kernels` = `--cpu-moe -ub 16` (CPU plain kernels) against the `-c 512` reference (GPU kernels); `kld-repack-vs-plain` = `--cpu-moe --no-host -ub 16` (`CPU_REPACK`) against `--cpu-moe -ub 16`.
- Here (CPU, random-weight 8-layer MoE with small embeddings so the softmax is not saturated): repack vs plain KLD 0.0039, but the same plain kernels at `-ub 8` vs `-ub 256` also give 0.018. On a random MoE any kernel change flips near-tied router choices, so these values say nothing about the size on Qwen3; only the measurement on the machine can rank the two causes.

## Task 7 - cold bucket in the file mapping by default

Since Task 6 the cold bucket has to be in a host buffer with the plain layout (for the prefill offload), so copying it had no benefit left: the kernels are the same as for the file mapping. The owner's run showed the cost of the copy: ~13.7 GB of cold experts copied into RAM for each placement run and ~50 s load, against 3-12 s for `--n-cpu-moe`, with 8.8 GB of swap already in use.

- With mmap (default) the cold bucket is the merged `ffn_{gate,up,down}_exps` tensor in the file mapping, as the experts of `--n-cpu-moe`: global ids, the experts of the other buckets skipped (skip flag) or mapped to expert 0 in the unflagged large-batch form. Only the hot experts are copied (to the GPU), plus the small id tables. The pages of the hot experts in the mapping are read once for that copy and are never used again; the OS can drop them.
- Without mmap (`--load-mode none`) the cold experts are copied into a host buffer as before.
- `--moe-ram-pin` only adds the locking now; the disk tier (unlocked cold experts left to the OS) is the default behavior with mmap.
- The loader line says which: `cold M experts (Y MiB, in the file mapping)` or `(Y MiB, copied, no mmap)`. With `-v` the placement buffers then hold only the id tables on the CPU side, and `CPU_Mapped` holds the cold experts.
- Verified (CPU and RPC devices, `-nr` so all runs use the same kernels): logits exact against the unsplit model at `-ub 512`, `16`, `1` for a hot set, a ranking, a ranking with `--moe-ram-pin`, all cold, and the hot set and all cold with `--load-mode none` (copy); on one RPC device (hot set, ranking with RAM pin, copy) and on two RPC devices (`--moe-devices`, three buckets); `regress-placement.py` (48 layers) on CPU and RPC.
- Note for CPU-only machines: `--cpu-moe` puts the experts into CPU_REPACK there (no host buffer type in the list), the placement's cold bucket uses the plain kernels, so exact checks need `-nr` on both sides. On the CUDA machine both use the plain CPU buffer of the file mapping.
- Not verified: load time and RAM use on the owner machine (expected close to `--n-cpu-moe`), CUDA offload at prefill from the mapping.

## Coding rules

- Minimal diffs in core files; prefer new files under `tools/`.
- Follow existing llama.cpp style (4 spaces, `snake_case`, no exceptions in
  hot paths).
- Every feature off by default; no behavior change for users who do not opt in.
- Small, focused commits with clear messages.
- At the end of each session, write a short summary: what was done, what was
  verified and how, what was NOT verified, open questions.

## Measured results - RTX 2060 Super 8 GB (2026-09-28)

Machine: RTX 2060 Super 8 GB (Turing, sm_75), 32 GB DDR4, Windows, MSVC + CUDA build.
Model: Qwen3-30B-A3B Q4_K_M (48 MoE layers, 128 experts, top-8). n_ctx 4096.
Conditions: clean run (RAG/Docker stopped), about 22 GB RAM free at start.
Full log: results/moe-20260928-015047/summary.md (not committed).

### Correctness

- all-cold placement vs --cpu-moe: exact (max abs diff 0) at ubatch 512 and 1.
- KLD vs --cpu-moe on wikitext-2 (40 chunks): 0.000000 for p3g, p5g and ranking,
  the same as --n-cpu-moe 40 (prefill offloads the cold bucket like the baseline).
- Kernel noise decomposition (ubatch 16, nothing offloaded):
  CPU vs GPU kernels 0.0094, repack vs plain 0.0086. The earlier 0.009 KLD of the
  placements was kernel noise, not a placement bug.
- At ubatch 1 one prompt flipped top-1 (max abs diff about 1.3), consistent with the
  kernel noise above on a near-tied token.

### Prefill (llama-batched-bench pp512, random tokens, median of 3)

| config | pp512 t/s |
|---|---|
| --n-cpu-moe 48 | 158 |
| --n-cpu-moe 40 | 169 |
| p3g | 162 |
| p5g | 183 |
| ranking (margin 1G) | 156 |

No prefill regression after the Task 6 fix.

### Decode on real text (llama-completion, 128 tokens, --temp 0)

Profile: tr_sr_pravni.csv + trace_sr.csv + trace_code.csv.
Both prompts are held out (not part of the profile traces).

| prompt | --n-cpu-moe 40 | p3g | p5g | ranking |
|---|---|---|---|---|
| code (src/llama-sampler.cpp) | 16.9 | 24.9 (+47%) | 29.6 (+75%) | 28.7 (+70%) |
| Serbian legal text, 1841 prompt tokens | 20.3 | - | 30.6 (+51%) | 29.3 (+45%) |

Serbian legal text prompt eval: 231 (ncmoe40), 229 (p5g), 216 (ranking) t/s.
Load time with placement (cold bucket in the file mapping): about 5-6 s.

### Where decode time goes (GGML_SCHED_TIMING=2, ranking, code prompt)

Per token: CPU cold experts about 20 ms, GPU about 15 ms. They run one after the
other within each layer, so the token time is close to the sum. With --n-cpu-moe 40:
CPU about 38 ms, GPU about 17 ms.

### Conclusions

1. Static per-expert placement gives +45-75% decode over --n-cpu-moe 40 on held-out
   in-domain text, with the same VRAM, the same prefill speed and identical output
   quality (KLD 0).
2. Out of domain (the English README fallback in the first run) the static hot set
   was worse than --n-cpu-moe 40. The profile must match the workload.
3. Next lever: overlap the GPU hot bucket and the CPU cold bucket within a layer
   (token time towards max(GPU, CPU) instead of the sum). This helps every domain.
   Phase 2 (dynamic residency) comes after that, depending on the remaining CPU time.

## Phase 2a decision data and a Windows note (2026-09-28)

### Real hit rates on held-out text (profile: tr_sr_pravni + trace_sr + trace_code, ranking budget 4.6 GiB)

| held-out text | static ranking | ideal LRU K=32 | realistic Phase 2a sim (section 8) | bytes/token | useful promotions |
|---|---|---|---|---|---|
| Serbian legal | 72.5% | 89.3% | 90.7% | 29 MiB | 77% |
| code | 66.3% (decode 63.9%) | 83.2% | 85.7% | 44 MiB | 74% |

Section 8 settings: warm start from the ranking, slot 0 pinned, second-miss admission
within 8 tokens, one-token commit delay, PCIe budget 12000 MB/s * 35 ms per token.
Both texts gain about 18-19 points over static placement, well above the 10-point gate
in PHASE2_REVIEW.md. Caveat: the simulation feeds prefill tokens one at a time.
Decision: implement Phase 2a.

### Windows run-to-run note

Same machine, same binaries, held-out Serbian legal text, decode t/s:

| run | --n-cpu-moe 40 | ranking |
|---|---|---|
| morning, HAGS off | 20.3 | 29.3 |
| after a reboot, HAGS off, warm file cache | 19.1 | 22.6 |
| after another reboot, HAGS on, warm file cache | 21.8 | 30.1 |

In the slow boot the GGML_SCHED_TIMING=2 profile showed the same per-layer kernel times
as the good runs, but the untimed run was no faster than the timed one, i.e. the
asynchronous GPU/CPU pipelining was lost. Cause unknown (not budget: 1720 hot experts in
all runs; not GPU clocks: no throttle reasons; power plan high performance).
HAGS is not proven to matter. Practical rules for measuring on Windows:
- after a reboot, warm the model file first (cmd /c "type <model.gguf> > nul");
- run each configuration at least twice and compare configurations within one boot;
- the ranking placement is more sensitive to this than --n-cpu-moe, because it has a
  CPU<->GPU split in every layer.

## Task 8 - Phase 2a, first milestone (2026-09-28)

Implements the Phase 2a MVP from PHASE2_REVIEW.md section 6/9 on top of Phase 1's static
placement: `--moe-dynamic` (opt-in, needs `--moe-placement` with a ranking, no
`--moe-devices`, mmap). Off by default; with it off, `llama_layer_moe_placement::dynamic`
stays false and the graph/loader path is exactly Phase 1's (no behavior change verified
below).

What it does, matching PHASE2_REVIEW.md's design:
- warm start: the hot bucket's slots keep their Phase 1 ranking assignment at load
  (`llama_moe_placement::initial_hot`, ascending global id order); local slot 0 always
  ends up the lowest-global-id hot expert and is pinned (never chosen as an eviction
  victim), matching PHASE2_REVIEW.md Q3/Q5.
- miss accounting and second-miss admission: `build_moe_ffn` marks the per-layer flat
  selected-expert-ids tensor as a graph output when the layer is dynamic
  (`llm_graph_result::t_moe_ids`); `llama_context` reads it back after `graph_compute`
  and hands it to the residency manager at the next `process_ubatch` boundary, one
  ubatch delayed (PHASE2_REVIEW.md's "one-token commit delay"). A cold expert is
  admitted for promotion on its second miss within 8 boundary calls (`ADMIT_WINDOW` in
  `llama-moe-residency.cpp`, matching the measurement above), or on every miss under
  `LLAMA_MOE_DYNAMIC_FORCE_CHURN=1` (a debug env var to force churn deterministically in
  a short test, per PHASE2_REVIEW.md risk 6).
- promotion: a dedicated worker thread with its own `ggml_backend_t` for the hot
  device and a pinned (or plain CPU, when the device has no host buffer type - e.g.
  RPC) staging region copies the expert from the cold (mmap) tensor into the staging
  buffer, then `ggml_backend_tensor_set_async` + `ggml_backend_synchronize` into the
  victim slot, separate from the compute stream. No ggml core change: this is the
  existing public backend API (PHASE2_REVIEW.md Q2/Q5/Q6).
- eviction: table update only (bucket + cold ids entries of the victim, done
  synchronously on the main thread before the copy starts) - no data movement, the
  merged cold tensor already holds every expert (PHASE2_REVIEW.md section 16).
- commit: promotions and evictions are only ever written to the device tables at a
  `process_ubatch` boundary, after `ggml_backend_sched_synchronize`, and only once the
  worker's copy has completed (a `done` queue drained at the boundary).
- counters: hits, misses, promotions, useful promotions (a promoted slot that was hit
  at least once before its next eviction), bytes copied, logged via `LLAMA_LOG_INFO`
  (same as the rest of this module - needs `-lv 5`/`--log-verbosity 5` to show, since
  `common_log_default_callback` maps `GGML_LOG_LEVEL_INFO` to trace verbosity) from
  `llama_moe_residency`'s destructor, so they print however the process exits.

New file `src/llama-moe-residency.{h,cpp}` (~420 lines); small additions to
`llama-moe-placement.{h,cpp}` (capture the warm-start set, validate both buckets/mmap/
single device for a dynamic layer, keep the placement object alive after load instead
of freeing it), `llama-model.{h,cpp}` (`moe_dynamic()`/`moe_dynamic_initial_hot()`,
`llama_layer_moe_placement::dynamic`), `llama-graph.{h,cpp}` (`t_moe_ids` output
capture), `llama-context.{h,cpp}` (own the residency manager, boundary()/
collect_routing() calls), `include/llama.h` + `common/` (the flag and its plumbing).

### Verification (CPU, and an RPC device as the hot device - same pattern as Task 3-7)

- `test-backend-ops -b CPU -o MUL_MAT_ID`: 930/930 pass (no ggml change in this task,
  so unaffected, but re-run for the record).
- `tools/expert-trace/regress-placement.py --dynamic` (new `--dynamic` flag): builds a
  ranking where every expert's count roughly halves with its id, the same for every
  layer, so `common_moe_placement_resolve`'s greedy count/byte fill is a deterministic
  round robin over layers regardless of per-layer expert byte differences (this model's
  quantizer picks Q6_K for `ffn_down_exps` on some layers and Q4_K on others - about
  1.46x - which is enough to starve a layer's hot bucket under a naively "equal counts"
  ranking; the id-halving keeps a >=2x gap between id tiers, safely above that). This
  guarantees every layer has at least 2 hot slots (pinned + 1 evictable) at a small
  `--moe-vram-margin`, itself picked by probing the VRAM budget report at the ubatch
  size that will actually run. Ran at 4 and 48 layers.
  - `--moe-dynamic` vs the unsplit model (`-nr`, same kernels): exact (max abs diff 0)
    at `-ub 1` (decode; PHASE2_DESIGN.md/PHASE2_REVIEW.md scope Phase 2a to decode),
    with `LLAMA_MOE_DYNAMIC_FORCE_CHURN=1` and its counters checked to confirm
    promotions and useful promotions were both > 0 (not just exact-by-no-op): 433
    promotions, 54 useful, 49 MiB copied at 4 layers; 6468 promotions, 1434 useful,
    719 MiB copied at 48 layers, from an 87-token prompt processed one token per
    ubatch.
  - Static placement (`hot`/`rank`, unaffected by this task) still exact at `-ub 512`
    and `-ub 1`, on CPU alone and with the RPC device, confirming the opt-in flag is a
    true no-op for everyone who does not pass it.
  - `-ub 512` (prefill, one ubatch for the whole prompt) was not run for `--moe-dynamic`
    on this tiny model: its compute-buffer size (hundreds of MiB, for KV cache/graph
    overhead unrelated to the ~1-10 MiB of tiny toy experts) swamps any small VRAM
    budget that would leave part of the hot bucket cold, so no ubatch-size-shared
    margin gives a partial hot/cold split at both `-ub 512` and `-ub 1` on this model.
    A real model's expert weights are orders of magnitude larger than this gap, so this
    is a tiny-test-model artifact, not a Phase 2a limitation - and prefill correctness
    for the static hot/cold split it inherits was already covered by Task 3-7's checks.
- Race review: PHASE2_REVIEW.md invariant 1 ("GPU never executes from a slot that is
  being overwritten") needed one more table write I had missed: on eviction, resetting
  the victim's `ids_hot` entry back to 0 (the same dummy every other cold expert uses),
  not just its `bucket`/`cold_ids*` entries. Without it, a graph that selects the
  just-evicted expert again before its next promotion would still read the victim slot
  for the discarded one-slot-per-row computation (PHASE2_REVIEW.md section 10/Q3),
  racing with the worker thread about to overwrite that slot for someone else. CPU/RPC
  testing cannot exercise this (their `set_tensor_async` falls back to a synchronous,
  already-serialized set), so it would not have shown up as a logit mismatch here; it
  is a correctness requirement for the real CUDA path. Fixed before verification above.

### Known simplifications in this milestone (not correctness issues, listed for the next one)

- Routing readback (`llama_context::collect_routing`) uses a synchronous
  `ggml_backend_tensor_get` right after `graph_compute` returns, not the async D2H +
  read-at-next-boundary PHASE2_REVIEW.md Q8 describes. Simpler and still correct (it is
  a few KiB), but adds one extra host sync per ubatch instead of zero, when dynamic
  residency is on. Swappable for the async version later without changing the
  interface.
- No admission rate limit (max experts in flight, byte budget per token) from
  PHASE2_REVIEW.md Q9/section 13 beyond the second-miss window and one promotion in
  flight per slot: a single worker thread processes the queue serially, which already
  bounds concurrency to 1, but does not bound PCIe/RAM bytes per token. Worth adding if
  real measurements show thrashing.
- Admission applies at every ubatch size, not decode-only as PHASE2_REVIEW.md section 9
  scopes it (prefill only seeds). Does not affect correctness (commits are always at a
  synchronized boundary regardless of ubatch size); it may promote experts during a
  large prefill batch that a decode-only policy would not have. Left as is for this
  milestone; worth gating on `ubatch.n_tokens == 1` alongside prefill seeding later.
- `ADMIT_WINDOW` (8) is a compile-time constant, not a CLI flag, to keep this
  milestone's surface small; matches the value already measured in the decision data
  above.
- The residency manager also gets constructed (and destructed) for the `no_alloc` dry
  runs `common_fit_params`/`common_moe_placement_resolve`'s probe step make before the
  real load (harmless - all-zero counters, confirmed above - but spins up a worker
  thread and a staging buffer for a throwaway context). Not measured; likely negligible
  next to a real model's load time, but worth avoiding if it shows up.

### Still needs the owner's GPU (not verifiable in this CPU-only sandbox)

- CUDA correctness: exact logits vs `--cpu-moe` are only verified with identical
  kernels (CPU/RPC); PHASE2_REVIEW.md's own correctness gate is exact on CPU/RPC, KLD
  on CUDA, same as Task 3-7. Also re-run `regress-placement.py`, ideally with a wider
  VRAM margin sweep (a small tiny-model budget was needed here only because of the
  probe workaround above, not a real constraint on real hardware).
- Whether the pinned staging buffer (`ggml_backend_dev_host_buffer_type`) and the
  second `ggml_backend_t` on the same CUDA device actually overlap the promotion copy
  with compute under WDDM, and whether `cudaHostRegister` on the mmap'd staging source
  would help (PHASE2_REVIEW.md Q6/risk 3) - both need real timing.
- Speed: decode t/s of `--moe-dynamic` vs the Task 6 ranking baseline, in and out of
  domain, per PHASE2_REVIEW.md section 22's matrix; the realistic hit rate vs the
  simulated one in the decision data above; whether the RAM-bandwidth contention risk
  (PHASE2_REVIEW.md risk 1, staging copy + cold FFN sharing RAM bandwidth) is real.
- Whether the extra host sync per ubatch from the synchronous routing readback
  (see "known simplifications") is measurable, before deciding whether to switch it to
  async.

Not started: Phase 2b (predictor/intra-token prefetch) and the GPU/CPU overlap work,
per the instruction to keep this milestone to Phase 2a only.

## Task 9 - Phase 2a CUDA measurement, hit-rate bug, admission diagnostics (2026-09-28)

First CUDA numbers (RTX 2060 Super, `heldout_code.txt`, 2077-token prompt + 128 decode):

| | static ranking | --moe-dynamic |
|---|---|---|
| pp t/s | 208 | 149-161 |
| tg t/s | 27.5 | 22.0-24.9 |
| counters | - | hits 73353, misses 756759, promotions 10157, useful 3723, bytes 28989554688 |

Slower than static, and the hit counter (8.8%) did not match the ~66% static hit rate
measured on the same text (WEIGHT_PROVIDER.md "Phase 2a decision data").

### Root cause of the hit-rate bug

`boundary()` processed one ubatch's routing as a single pass, in order: for each
selected expert, classify hit/miss against `expert_slot`, *then* immediately decide
admission and evict if admitted, which flips `expert_slot` for the victim right away.
The 2077-token prompt is one large prefill ubatch (default `-ub` 2048/512), so all of
its routing is processed inside one `boundary()` call, at one tick. Two failure modes
followed from that:

1. **Every expert looked like "a second miss inside the window" immediately.** The
   admission window (8 ticks) was counted in `boundary()` calls, not tokens; with an
   entire 2077-token prefill inside one tick, any expert selected twice anywhere in the
   prompt (nearly all of them, on a top-4/128 router) qualified for promotion right
   away. That is the promotion storm (10157 promotions, ~29 GB, matching the report).
2. **Hits got undercounted, badly.** Because eviction mutated `expert_slot` while the
   same pass was still classifying later entries of the same ubatch, an expert that
   was genuinely hot when the graph actually ran could be evicted by an earlier entry
   in the routing list before its own (later) entry was checked - counted as a miss
   even though the graph read it hot. On a 2077-token prompt with a small hot set this
   happened constantly, which is why the counted hit rate (8.8%) was far below the
   real one (~66%): most of the "misses" were bookkeeping artifacts, not real cache
   misses at the time the graph ran.

Both are `boundary()` bugs, not PCIe/RAM contention. Fixed by splitting `boundary()`
into two passes per layer: pass 1 classifies every routed id as a hit or miss against
a state that pass 1 itself never mutates (LRU-touching a hit is fine, it does not
change hit/miss classification of other entries), collecting eligible misses into a
deduplicated candidate list; pass 2 runs `admit()` (which does mutate `expert_slot`
and the device tables) only for that list. This alone fixes the hit-count corruption
regardless of ubatch size, and is not behind a flag - it is a bug fix, not a policy
change.

### Admission policy switches (each off by default, reproduces the base moe_dynamic behavior)

- `--moe-dynamic-batch-threshold N`: no admission from ubatches with more than N
  tokens (their hits/misses are still counted). Suggested: 32, the `--n-cpu-moe` / op
  offload batch size.
- `--moe-dynamic-decode-window`: count the admission window in actual ubatch tokens
  (`tick += ubatch.n_tokens`) instead of one tick per ubatch regardless of size. Fixes
  failure mode 1 directly; independent of the threshold switch (e.g. useful together
  with a *raised* threshold that still allows admission from moderate-sized batches,
  windowed correctly).
- `--moe-dynamic-bw MB/s`: cap bytes admitted for promotion at one boundary to this
  many MB/s times the wall-clock time since the previous boundary that was eligible to
  admit (a real per-step rate limit, not just a window/threshold heuristic).

Hit/miss counters are now always split prefill/decode (`ubatch.n_tokens > 1` = prefill;
this classification does not depend on the switches above), and there are now separate
decode-only derived numbers: bytes copied per decode token, and useful-promotion %.

### Per-promotion event log and classifier (env `LLAMA_MOE_DYNAMIC_LOG=<path.csv>`, opt-in)

Added before changing the default admission policy, so the current (unfixed-policy,
bug-fixed-counters) behavior can be diagnosed directly rather than guessed at. One row
per promotion, written by `llama_moe_residency::write_promo_log()` at destruction; see
the field comments on `llama_moe_residency::promo_event` in
`src/llama-moe-residency.h` for exactly what each column means and when it is set
(admission, transfer start/end, commit, first reuse and reuse count, eviction, and -
after eviction - whether it was missed again).

`tools/expert-trace/promo-report.py` reads the CSV and reports, split prefill/decode:
hit/reuse/eviction summary stats; the reuse-count distribution (0, 1, 2, 3, 4-7, 8+);
a net-benefit estimate per promotion (`reuse_count * --saved-ms-per-reuse -
bytes / (--transfer-mbs * 1000)`, both given as arguments, defaults 0.065 ms and 12000
MB/s) and how many promotions were net positive; and a classification of every
non-useful promotion (reuse_count == 0) into one of six likely causes (threshold too
low, evicted too early, transfer too late, no longer needed, queue delay, slot
pressure) - see the script's docstring for the exact rule per class and the priority
order between them.

### Verification (CPU, RPC device as the hot device)

- `test-backend-ops -b CPU -o MUL_MAT_ID`: 930/930 (unaffected by this task, re-run for
  the record).
- `regress-placement.py --dynamic`, extended: the `dyn` run (all switches off,
  reproducing the base/current policy) and a new `dyn_sw` run
  (`--moe-dynamic-batch-threshold 32 --moe-dynamic-decode-window --moe-dynamic-bw
  12000`, i.e. all three switches on together) are both exact (max abs diff 0) against
  the unsplit model at `-ub 1`, with `LLAMA_MOE_DYNAMIC_FORCE_CHURN=1` and both runs'
  counters checked for promotions > 0 and useful promotions > 0. At 4 layers: `dyn` 432
  promotions/140 useful, `dyn_sw` 421/133. At 48 layers: `dyn` 6425/2975, `dyn_sw`
  6361/2903. The switches change *how many* promotions are admitted and when, never
  correctness of a promotion once admitted, so exactness holding with them on is the
  expected result, not evidence either way about their effect on the hit rate (that
  needs the owner's GPU, see below).
- `promo-report.py` smoke-tested against two logs: a 3-row log from a single-shot
  `-ub 512` `llama-debug` run (only a warmup ubatch's promotions ever reach a
  `boundary()` call in that single-decode tool, so it cannot commit anything from the
  real prompt - not a representative dataset, but exercises the parser/classifier on
  real "admitted, never committed" rows) and a 441-row log from the same `-ub 1`
  `LLAMA_MOE_DYNAMIC_FORCE_CHURN=1` setup used for the exactness checks above (32.0%
  useful, reuse-count distribution 0:300/1:141, net benefit +5.0 ms, 96% of non-useful
  promotions classified `F` slot pressure - expected, this tiny test model has only
  2-3 hot slots per layer under forced immediate admission). Both ran without error and
  produced numbers consistent with the underlying counters.
- Did not verify the *decode-token accuracy* of `--moe-dynamic-batch-threshold` /
  `--moe-dynamic-decode-window` against a real prefill-then-decode sequence in this
  sandbox: `llama-debug` (used for the exact-logit checks) makes exactly one
  `llama_decode()` call, so a `boundary()` call only ever exists to apply *previously*
  queued promotions, never to admit-and-commit from the same run's own prefill - a
  structural property of that tool, not of the fix. `llama-cli`'s real generation loop
  needs a VRAM-budget probe of its own (its context defaults differ enough from
  `llama-debug`'s that the probed margin above did not carry over) that there was not
  time to add here; manually confirmed instead, with a single-shot `-ub 512` run
  against the same model, that the promotion log correctly labels admissions
  `phase=prefill` when a genuine multi-token ubatch triggers them (see the 3-row log
  above). The owner's CUDA run is real prefill-then-decode and is the real test of the
  threshold/window switches' effect on the hit rate; see below.

### Still needs the owner's GPU

- Re-run the CUDA measurement above (static ranking vs `--moe-dynamic`) with the fixed
  hit-rate counters alone (all switches still off) to get an accurate baseline hit rate
  first, then with `--moe-dynamic-batch-threshold 32 --moe-dynamic-decode-window`
  (fixes both prefill-storm failure modes) and separately with `--moe-dynamic-bw 12000`
  added, to see which switch combination, if any, recovers the static ranking's
  decode t/s while keeping the out-of-domain adaptivity Phase 2a is for.
- `LLAMA_MOE_DYNAMIC_LOG` + `promo-report.py` on that same real run: the class
  breakdown (A-F) should say directly whether the remaining non-useful promotions (if
  any, once the switches are on) are mostly prefill artifacts (class A, should drop to
  ~0 with the threshold switch), genuinely evicted too early (class B, LRU may need
  frequency or a longer window), or slot pressure from too few hot slots for the
  working set (class F, a budget question, not a policy one).
- Net benefit at real (not force-churn) timings: rerun `promo-report.py` with
  `--saved-ms-per-reuse` set from the actual measured CPU-vs-hot per-expert time
  difference on that machine, not the current placeholder default.
- Everything already listed as needing the GPU in Task 8 above (CUDA exactness/KLD,
  WDDM overlap, RAM-bandwidth contention) still applies unchanged.
