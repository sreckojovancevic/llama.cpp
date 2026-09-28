# Phase 2 design review (dynamic MoE expert residency)

Review of `PHASE2_DESIGN.md` against the code on branch `wp-task3` (Phase 1, Tasks 3-7) and the RTX 2060 Super results in `WEIGHT_PROVIDER.md`. No code was changed. Line numbers refer to this branch.

Section 28 of the design has 12 questions. "Question 13" is taken here as the ordering question raised by conclusion 3 of the measured results: overlap of the hot and cold buckets first, or Phase 2a first.

## Summary

- **Feasible, and cheaper than the design assumes.** Phase 1 already has the indirection that Phase 2a needs: per-layer I32 tables (`ids_hot`, `ids_cold`, `ids_cold0`, `bucket`) that the graph reads on the device at run time. The hot bucket tensor `[.., n_hot]` per layer is already a set of fixed VRAM slots. Dynamic residency means rewriting slot contents and table entries between graph runs. The graph topology, tensor pointers, llama graph reuse and CUDA graphs do not change.
- **Eviction needs no copy.** With mmap the cold bucket is the full merged `ffn_*_exps` tensor in the file mapping, addressed by global ids (Task 7). Every expert is always available on the CPU, so eviction only changes table entries. The "VRAM -> CPU eviction" transfer in design section 18 is not needed.
- **Phase 2a needs no ggml changes.** It can use the public ggml API: a second backend instance for the transfer stream, `ggml_backend_tensor_set_async`, `ggml_backend_synchronize` and `ggml_backend_tensor_set`. All changes stay in `src/` and `common/`, plus a worker thread.
- **The overlap gain is smaller than conclusion 3 says.** Only the hot FFN GPU work of a layer can run alongside that layer's cold FFN. Attention, router and combine stay on the dependency chain, so the token time does not drop to max(GPU, CPU). My estimate is 2-3 ms of about 35 ms per token (+6-9%). Overlap also needs changes to the ggml scheduler.
- **Phase 2a targets the larger cost.** The owner's numbers suggest the real VRAM hit rate on held-out code was about 58%, not the 70.8% profile share (derivation in Q13). The CPU cold FFN (about 20 ms per token) is then the largest item, and LRU in the same VRAM simulates at 82-89%.
- **Recommendation:** first spend 1-2 days measuring and extending the simulator (numbers only, no core code). Then build Phase 2a as an online extension of the Phase 1 ranking placement. Do overlap after that; it gains relatively more once the CPU share has shrunk. Details are in "Recommendation".

## 1. What the code does today (facts used below)

Expert storage and graph (Phase 1):

- `llama_layer_moe_placement` (`src/llama-model.h:255-268`) holds a list of `llama_moe_bucket {gate, up, down, ids, ids0, skip}` plus one `bucket` table `I32 [1, n_expert]`.
- Loader (`src/llama-moe-placement.cpp:130-315`):
  - Hot buckets are new tensors `[.., n_hot]` in one buffer per device (`alloc`, `:317-341`).
  - Hot slices are copied from the file once (`load`, `:343-376`). After that the loader drops the host copies of the tables (`tables.clear()`).
  - With mmap the cold bucket is the merged tensor itself (`:244`) with global ids (`:178`). Without mmap only the cold experts are copied, with local ids.
  - A layer with no hot or no cold experts gets fewer buckets (`:159-184`), which changes the graph shape.
- `build_moe_ffn` (`src/llama-graph.cpp:2335-2391`):
  - `ids_flat = cont(selected_experts)`, then `get_rows(table, ids_flat)` per bucket.
  - The hot bucket uses the one-slot-per-row form (`:2360-2365`). Slots of experts in other buckets get local id 0, so **hot slot 0 is read on every token** and its rows are dropped by the final `get_rows(stacked, bucket)` (`:2379-2384`).
  - The cold bucket runs flagged (`skip`, ids -1) below `GGML_OP_OFFLOAD_MIN_BATCH`, and unflagged with `ids0` at or above it (`:2350-2358`).

Scheduler (`ggml/src/ggml-backend.cpp`):

- Splits run strictly in order in `ggml_backend_sched_compute_splits` (`:1668-1892`).
- Events exist only with pipeline parallelism, `n_copies > 1` (`:1936`, `:1969-1973`). That needs more than one device with all layers offloaded and no tensor overrides (`src/llama-context.cpp:428-455`), so it is off in the owner's setup.
- Before a CPU split that reads a CUDA tensor, `cpy_tensor_async` is not available, so the scheduler calls `ggml_backend_synchronize(input_backend)` on the whole CUDA stream, then copies (`:1803-1813`). **This is the per-layer host sync point at decode.** It is also why the hot FFN, if it sits in the same GPU split as the router, must finish before the CPU cold FFN can start.
- For an offloaded MUL_MAT_ID with host weights, the scheduler reads the ids back (sync) and copies only the used experts, synchronously and per ubatch (`:1713-1801`). This is the existing "demand paging" path, used at prefill.
- Graph inputs (`GGML_TENSOR_FLAG_INPUT`) are assigned to the last backend, the CPU (`:955-956`), and copied to each split that uses them.

CUDA backend (`ggml/src/ggml-cuda/`):

- Each `ggml_backend_cuda_context` creates its own non-blocking streams on first use (`common.cuh:1528-1534`). A second backend instance on the same device therefore gets its own stream.
- `set_tensor_async` is a `cudaMemcpyAsync` on the backend's stream (`ggml-cuda.cu:2445-2452`). The synchronous buffer `set_tensor` uses `cudaStreamPerThread` and then synchronizes (`:784-790`).
- Event record and wait use `cudaEventRecord` and `cudaStreamWaitEvent` (`:4479-4489`). The public event API is new/free/record/synchronize/wait (`ggml/include/ggml-backend.h:124-128`); **there is no non-blocking query**.
- CUDA graphs are keyed by the first node of the split. They are re-captured only when node properties, `src` data pointers, `ne` or `nb` change (`:2595-2635`), never when data changes. Graphs are disabled for a MUL_MAT_ID that needs a host sync (`:1881-1908`, `:2558-2589`); `n_experts = src0->ne[2]` (slot count) enters the MMQ choice.
- The CUDA device reports `buffer_from_host_ptr = false` (`:5041`). `ggml_backend_cuda_register_host_buffer` exists behind `GGML_CUDA_REGISTER_HOST` but llama.cpp does not call it (`:4838-4859`).

llama context (`src/llama-context.cpp`):

- `process_ubatch` (`:1336-1406`) either reuses the previous graph (`can_reuse`, `:1350`) or rebuilds it. It then calls `set_inputs` and `graph_compute`, which calls `ggml_backend_sched_graph_compute_async` (`:2514`).
- `decode` starts output reads with `ggml_backend_tensor_get_async` (`:1886`) and **returns without synchronizing**. The sync comes later, from `llama_get_logits*` / sampling (`llama_context::synchronize`, `:716`).

## 2. Answers to the section 28 questions

### Q1. How does llama.cpp represent MoE expert tensors and their storage?

- Upstream stores one merged 3D tensor per projection and layer: `ffn_{gate,up,down}_exps [n_embd, n_ff_exp, n_expert]`, with expert `e` at byte offset `e*nb[2]`. MUL_MAT_ID picks experts through an I32 ids tensor `[n_expert_used, n_tokens]`.
- The tensor lives in one backend buffer (CUDA, CPU, `CPU_REPACK`, pinned host, or the file mapping with mmap). `-ot` / `--n-cpu-moe` only choose that buffer per tensor.
- Phase 1 adds per-layer buckets. The hot bucket is a smaller tensor of the same row layout with local ids. The cold bucket is the merged tensor in the file mapping with global ids. The mapping is done by `get_rows` on small I32 tables that sit on the device (section 1).
- Each tensor's `data` pointer is fixed after load. The scheduler, the allocator, views and CUDA graphs all rely on that.

### Q2. Where is it safe to add residency indirection?

- **In the existing I32 tables, not in tensor pointers.** A global -> slot mapping is already evaluated on the GPU at run time: `ids_hot` (global -> hot slot), `bucket` (global -> which row to take), and `ids_cold` / `ids_cold0` (global -> cold id, or -1 / 0 when hot). Changing their contents changes which slot is read, with no change to graph topology, pointers or shapes. llama graph reuse (`can_reuse`) and CUDA graphs therefore stay valid (`ggml-cuda.cu:2616-2631` compares pointers and shapes, not contents).
- **Not safe:** changing `tensor->data` or `buffer` of expert tensors. That would invalidate CUDA graph captures, break the scheduler's hash of tensor copies, and the MUL_MAT_ID copy path (`ggml-backend.cpp:1713-1801`) assumes stable addresses.
- **Keep the tables as weights, not graph inputs.** Graph inputs are placed on the CPU and copied to every split that uses them (`ggml-backend.cpp:955-956`, `:1697-1704`). That is 3-4 small H2D copies per layer per token, 150-200 extra `cudaMemcpy` calls, likely 1-2 ms per token on WDDM. As weights they are updated only when residency changes.
- To make each commit a single copy, put all tables of all layers in one contiguous tensor (for example `I32 [n_expert, 4, n_layer]`, with per-layer views) and keep a host mirror.

### Q3. How do fixed VRAM slots fit the ggml tensor execution model?

- **Slots = the hot bucket tensor of each layer.** Slot `s` of layer `L` is the byte range `[s*nb[2], (s+1)*nb[2])` of `blk.L.ffn_{gate,up,down}_exps.hot`. Writing an expert into a slot is three `set_tensor_async` calls at that offset, one per projection.
- **Per-layer slot count K_L is fixed at load.** It can come from the Task 4 ranking budget, i.e. the current `n_hot` per layer. A global pool shared by all layers (one tensor, per-layer tables pointing into it) is possible because Qwen3 layers have identical shapes. It would let the split between layers adapt, but it changes `n_experts` of every hot MUL_MAT_ID to the pool size, which affects the CUDA kernel choice (`should_use_mmq(..., n_experts)`). Start with per-layer slots.
- **In dynamic mode every layer must have both buckets.** Without that, a layer that has no hot experts at load time has a different graph shape (`llama-moe-placement.cpp:159-184`). K_L >= 1 for every layer, and the cold bucket is the full merged tensor.
- **Slot 0 of each layer must never be rewritten** (see Q5): the one-slot-per-row form reads it for every non-hot slot. Pin the hottest ranking expert of the layer there. It is still a normal hot expert, just never evicted.
- **Dynamic mode needs mmap** (the default), or a copied cold bucket that contains all experts. With `--load-mode none` today only the cold experts are copied, so an evicted expert would have no CPU copy.

### Q4. At which exact token/layer boundary can the residency table change?

- **Boundary for Phase 2a: the start of `llama_context::process_ubatch`, after `ggml_backend_sched_synchronize`.** That is the only point where the host knows no graph is running (`src/llama-context.cpp:1336-1396`).
- `llama_decode` returning is **not** that point: it returns with GPU work and async output reads still in flight (`:1886`). In normal decode the sampler's `llama_get_logits*` has already synchronized, so the explicit sync is almost free.
- It is a ubatch boundary, not strictly a token boundary. At decode (`n_ubatch = 1`) they are the same. At prefill the policy should not promote per ubatch, because prefill offloads the cold bucket anyway (`llama-graph.cpp:2350-2351`); it can only record routing counts.
- **Layer boundaries inside a graph:** at decode the scheduler synchronizes the CUDA stream on the host before every CPU split (`ggml-backend.cpp:1806`). So there is a host-visible idle point per layer, but no hook to run code there. Using it would need a scheduler callback between splits, which is a Phase 2b item (Q11).

### Q5. How do CUDA events guarantee safe slot reuse?

With the design below, Phase 2a needs **no GPU-side event waits**. Host-side ordering is enough:

1. **Evict** (boundary B0): after `sched_synchronize`, flip the victim's table entries to cold (`bucket = cold`, `ids_cold = e`, `ids_cold0 = e`) with one synchronous `ggml_backend_tensor_set`. No graph that starts after this reads slot `s`.
2. **Copy:** after that, and only then, the worker writes the new expert into slot `s` on the transfer stream and ends with `ggml_backend_synchronize(transfer_backend)`, i.e. `cudaStreamSynchronize`. Then it sets an atomic "done" flag.
3. **Promote** (a later boundary B1): the main thread sees "done", flips the entries of the new expert to hot and issues the next graph. The host observed the copy's completion before it launched that graph, so the compute stream sees the finished data.

Two copies of this ordering are not allowed:

- Writing a slot that the tables of the running graph still reference (invariant 1): prevented because eviction commits before the copy starts, and B0 is after a full sync.
- Slot 0: always referenced (Q3), so it is never a victim.

`cudaEventQuery` is not exposed by ggml (`ggml-backend.h:124-128`). The design's "CUDA event -> transfer completed?" check at the boundary would need either a new ggml API (`ggml_backend_event_query`) or the worker-thread pattern above. Use the worker thread: no ggml change, and the main thread never blocks on a copy. Events are only needed for Phase 2b, where commits happen without a host sync.

### Q6. How can a transfer stream be separate from the compute stream?

- **Create a second backend instance for the same device** with `ggml_backend_dev_init(dev, nullptr)`. It gets its own `ggml_backend_cuda_context` and its own non-blocking stream (`common.cuh:1528-1534`). The worker thread owns it exclusively; ggml backend objects are not thread-safe, so no sharing.
- `ggml_backend_tensor_set_async(transfer_backend, slot_tensor, src, offset, size)` works because the slot tensor is in a CUDA buffer of the same device (`ggml-cuda.cu:2449`).
- **The source must be pinned for a truly async DMA.** The file mapping is pageable, and the CUDA device has no `buffer_from_host_ptr`. So the worker:
  1. copies the expert from the mapping into a small pinned staging buffer (`ggml_backend_dev_host_buffer_type`, 2-4 experts, double buffered). Any page faults of the disk tier happen here, in the worker, not in the main thread;
  2. issues the three async H2D copies;
  3. synchronizes the transfer stream.
- Alternative to test: `cudaHostRegister` on the mapped range (the code exists behind `GGML_CUDA_REGISTER_HOST`). It removes the CPU memcpy, but registering a file-backed mapping may fail, especially on Windows.
- Turing has separate copy engines, so the H2D copy overlaps the compute stream's kernels.

### Q7. How does CPU fallback connect to the Phase 1 path?

- **It already is the Phase 1 path, unchanged.** An expert whose `bucket` entry says cold is computed by the cold MUL_MAT_ID on the CPU, from the file mapping, with its global id: flagged at decode, offloaded at prefill.
- A miss therefore needs no special handling. It is simply the expert's current table state. Promotion failure, a copy still in flight, or a copy that never finishes all leave the expert cold, which satisfies invariants 4 and 5 by construction.
- Correctness does not depend on the policy at all: any table state that is consistent (each expert in exactly one bucket, slot contents matching `ids_hot`) gives the same result as Phase 1 with that hot set. On CPU/RPC that is exact, as the existing checks show.

### Q8. Where can the request/promotion queue live without touching the correctness path?

- **On the host, in a new residency manager owned by `llama_context`,** that only writes the tables at the boundary. The graph and the kernels never see it.
- **Routing information must come back to the host.** Today it is only visible through `cb_eval`, which synchronizes per tensor and is far too slow. Proposal:
  - `build_moe_ffn` copies `ids_flat` of each layer into a view of one persistent I32 "routing log" tensor on the GPU, `[n_expert_used*n_ubatch, n_layer]`. This is the same pattern as KV cache writes: `ggml_cpy` into a view of a tensor outside the graph allocator.
  - After each ubatch, one `ggml_backend_tensor_get_async` copies the log to the host. It is read at the next boundary, after the sync.
  - Cost: 48 tiny copy kernels plus one D2H copy of 1.5 KiB per token at decode.
- **Dedup is trivial:** there is one state per (layer, expert): `COLD`, `IN_FLIGHT(slot)`, `HOT(slot)`. The design's per-request objects (section 11) and their dedup (section 12) collapse to "admit only when COLD". Keep counters, not request objects.
- The queue itself is a bounded FIFO of `{layer, expert, slot}` jobs from the main thread to the worker, plus a completion list back. Both are protected by a mutex; at most a few hundred entries per second.

### Q9. What is the safest minimal LRU / admission model?

Per layer:

- **LRU over slots** (with slot 0 pinned). The recency of every hot expert is updated from the routing log on every token, hits included.
- **Admission: second miss within a window.** A cold expert is admitted when it was selected at least twice within the last W decode tokens (W = 8-16, one `last_miss` timestamp per (layer, expert)). This is the design's section 13 rule.
- **Victim:** the least recently used HOT expert of the layer that was not used in the last token and is not pinned. If none qualifies, do not promote.
- **Rate limits:**
  - at most N experts in flight in total (N = 4-8);
  - a promotion byte budget per token, for example 50% of `PCIe bandwidth x measured token time`; the rest is dropped (the expert stays cold, nothing is lost).
  - This bounds PCIe and RAM traffic, which are the thrashing risks.
- **Warm start:** the initial slot contents are the Task 4 ranking hot set. With the admission threshold at infinity the system is exactly Phase 1. That gives a clean A/B switch and a safe fallback: "Phase 2a <= Phase 1 in no domain" becomes a policy-tuning question, not a correctness question.
- **Prefill seeding (cheap and probably valuable):** routing during the prompt tells which workload is running (the cross-workload Jaccard is 0.04). At the end of prefill, admit the prompt's most frequent cold experts under the same budget. It is a one-shot adaptation per request.

Only the LRU part has been simulated, with instant insertion. The admission rule, the one-token commit delay and the byte budget are not simulated yet (see Recommendation).

### Q10. What are the concrete obstacles to Phase 2a?

1. **Routing is not on the host:** needs the routing log (Q8). Small graph change in `build_moe_ffn`.
2. **Graph shape depends on the load-time hot set** (empty buckets are removed): dynamic mode must force both buckets in every layer (Q3).
3. **Slot 0 is read on every token** as a dummy (Q3/Q5).
4. **Pageable source:** a pinned staging buffer and a worker thread are needed (Q6). The staging copy costs RAM bandwidth, which the CPU cold FFN also needs; it is already near the limit at 37-44 GB/s (Task 6). This is the main performance risk (see Risks).
5. **No non-blocking event query in ggml:** the worker thread avoids it (Q5).
6. **The loader throws away table data and slot metadata after load** (`llama-moe-placement.cpp:375-376`). A persistent descriptor is needed (tensors, `nb[2]`, merged source tensors and mapping pointers, host table mirror).
7. **Scope limits for the first version:** one hot device (no `--moe-devices`), mmap only, `qwen3moe` only, decode policy only, no LoRA or expert scales. These match the existing Phase 1 limits.
8. **The measurement gap:** the 82-89% LRU number assumes instant insertion, no admission and no bandwidth limit. The real 2a hit rate will be lower; how much lower is unknown.

None of these needs ggml changes.

### Q11. Does Phase 2b require graph scheduling changes?

Yes. Intra-token prefetch means a promotion becomes visible within the same graph, at a layer boundary. Options:

- **Host hook between splits:** add a callback to `ggml_backend_sched_compute_splits`, called before a split with the split's first node. At decode the host already waits for the GPU before each CPU split (`ggml-backend.cpp:1806`), so a table update there is safe. But the host work then sits on the critical path of every layer, and it only works where such a sync exists.
- **Device-side commit:** the transfer stream records an event and writes a device "ready" flag. A small custom op before layer L's `get_rows` chooses between the old and new table from that flag. This needs a new ggml op or CUDA-only code, careful memory ordering across streams, and it must be captured in CUDA graphs. A `cudaStreamWaitEvent` would violate "the GPU never waits".
- **The predictor itself** (hidden state of layer L -> experts of L+1) adds nodes whose output the host needs during the graph, which again means a sync or device-side logic.

Also, events in the scheduler exist only for `n_copies > 1`; 2b would need them in the single-GPU case too. So 2b is a ggml core change, and it should only be considered after 2a numbers exist, as the design already says.

### Q12. Which parts of llama.cpp / ggml would the implementation have to change?

See "Required changes per file" below. In short: Phase 2a touches `src/` (placement loader, graph, context, a new residency module) and `common/` (arguments), with no ggml change. Overlap touches `ggml/src/ggml-backend.cpp` (scheduler). Phase 2b touches the scheduler and the CUDA backend.

## 3. Q13: overlap first or Phase 2a first?

### What overlap can gain

The per-layer dependency chain at decode is:

```
attn(L) + router(L) [GPU] -> { cold FFN(L) [CPU] || hot FFN(L) [GPU] } -> combine(L) [GPU] -> attn(L+1) ...
```

- Only `hot FFN(L)` can run in parallel with `cold FFN(L)`. The saving is at most `sum over L of min(hot_L, cold_L)`, which is at most the total hot FFN GPU time. It is not `sum - max(GPU, CPU)`: attention, router and combine (most of the 15 ms of GPU time) stay serial.
- Estimate of the hot FFN GPU time per token, ranking run:
  - about 270 hot expert FFNs (70% of 384) x about 2.7 MB (Q4_K_M expert of Qwen3-30B-A3B) = 0.73 GB at about 400 GB/s: about 1.8 ms;
  - plus about 8 extra kernels per layer (repeat, 3 MMVQ, swiglu, get_rows, concat, select): about 0.5-1 ms.
  - Total **about 2-3 ms of the measured 34.8 ms per token, so about +6-9%**.
- Indirect support for this estimate: `--n-cpu-moe 40` (64 expert FFNs on the GPU) measured 17 ms of GPU time and the ranking (about 270) measured 15 ms. So the GPU time hardly depends on the number of hot experts, and most of it is per-layer work and overhead, not the hot FFN.
- Cheap check: GPU time of `--n-cpu-moe 48` (no experts on the GPU) against the ranking with `GGML_SCHED_TIMING=1`. The difference is roughly the hot FFN time, i.e. the overlap ceiling.

### What overlap costs

- **Scheduler change in ggml core.** Today the CPU split first synchronizes the whole CUDA stream (`ggml-backend.cpp:1806`). Overlap needs:
  1. graph order `[attn, router, ids] GPU -> [cold] CPU -> [hot] GPU -> [combine] GPU`;
  2. a split boundary between hot and combine, which requires a scheduler rule, because consecutive GPU nodes merge into one split;
  3. "launch-ahead" in `compute_splits`: issue the async GPU hot split before running the blocking CPU split, when it does not depend on it.
- This is roughly 200-400 lines in `ggml-backend.cpp` plus tests. The code is shared by all backends and by pipeline parallelism, and allocator reuse across splits must be rechecked (`:1681-1689`). It is the kind of change upstream is careful with, and it is hard to test without a GPU in the sandbox.

### What Phase 2a can gain

From the owner's numbers:

- With `--n-cpu-moe 40`: 38 ms CPU for 40 x 8 = 320 experts, about 116 us per expert including the per-layer base cost.
- Ranking: 20 ms CPU works out to about (20 - 48 x 0.025) / 0.116 = about 160 CPU experts per token. That is a **real VRAM hit rate of about 58%** on the held-out code text, against the 70.8% profile share. This is candidate 1 of Task 6, and it is only an inference from timings; trace the text to confirm it.
- LRU with a similar VRAM budget simulates at 82-89% (Phase 0). At 85% there would be about 58 CPU experts, about 7 ms instead of 20 ms, so the token time drops from about 35 ms to about 22-24 ms (+45-60%). That is before admission, commit delay and bandwidth losses; a realistic value might be half of it.
- Transfer volume at 85% hits: about 58 misses x 2.7 MB = about 150 MB per token (only admitted ones are copied). At 40 t/s that would be 6 GB/s over PCIe 3.0 x16 (about 12 GB/s pinned). It fits, but the staging copy and the DMA read also use RAM bandwidth that the CPU cold FFN needs. The admission rule and the byte budget exist for exactly this reason.
- Phase 2a also addresses conclusion 2 (the static hot set loses out of domain), which overlap cannot.

### Answer

**Phase 2a first, after a short measurement step. Overlap second.**

- Phase 2a has the larger expected gain and needs no ggml change. It degrades to exactly Phase 1 when switched off or when admission is disabled.
- Overlap gains more after Phase 2a: once CPU time falls toward the hot FFN time, `min(hot, cold)` is a larger share of a shorter token.
- Revisit this order if the measurement step shows that:
  - the real static hit rate is already above about 80%, so there is little CPU time left to remove; or
  - the simulated realistic 2a hit rate barely beats the static one; or
  - the hot FFN GPU time turns out to be large.

  In any of these cases overlap comes first.
- Separate observation: 15 ms of GPU time against about 4 ms of estimated GPU memory traffic suggests per-split overhead (96 splits per token, WDDM) is a large cost of its own. Neither item removes it. It is worth a look with Nsight Systems before either.

## 4. Conflicts between PHASE2_DESIGN.md and the code

| design section | conflict | resolution |
|---|---|---|
| 9, 17, 20 "token boundary" | `llama_decode` returns before the GPU finishes; the natural boundary is per ubatch | Commit at the start of `process_ubatch` after `ggml_backend_sched_synchronize` (Q4) |
| 9, 17 "CUDA event -> transfer completed?" | ggml has no non-blocking event query | Worker thread synchronizes its own stream and sets an atomic flag (Q5) |
| 10, 25 inv. 1 | Slot 0 of each layer is read on every token (one-slot-per-row dummy id 0) | Pin slot 0; never evict it (Q3) |
| 16, 18 "VRAM -> CPU eviction" transfer | Not needed: the cold bucket is the full merged tensor in the mapping | Eviction is a table change only |
| 7, 19 slot model | A layer with an empty bucket has a different graph shape | Dynamic mode: every layer has K >= 1 slots and the full cold bucket |
| 8 "residency table authoritative" | The graph reads device tables; a host table alone is not what runs | Host mirror = single writer; device tables are a snapshot written only at commit |
| 11, 12 request queue and dedup | Per-request objects are not needed | One state per (layer, expert); dedup = admit only when `COLD` |
| 6, 22 "82-89% LRU" | Simulated with instant insertion, no admission, no bandwidth limit | Re-simulate with commit delay, admission and byte budget before judging 2a |
| 13 admission | The design leaves the window and threshold open | Second miss within W = 8-16 tokens, plus in-flight and byte limits (Q9) |
| 18 transfer stream | Source is pageable mmap; async H2D needs pinned memory | Pinned staging buffer in the worker, or test `cudaHostRegister` on the mapping |
| 3 "GPU never waits" | Holds in 2a; would be broken by `cudaStreamWaitEvent` in 2b | 2b needs device-side ready flags or host hooks (Q11) |
| 22 baseline | Phase 1 has several variants (hot set, ranking, `--moe-ram-pin`) | Baseline = the same ranking file, with admission off; same VRAM |
| - (missing) | Prefill offloads the cold bucket; dynamic residency only matters for decode | Policy for decode only; prefill only seeds (Q9) |
| - (missing) | `--load-mode none` copies only the cold experts | Dynamic mode requires mmap, or a copy of all experts |
| - (missing) | `--moe-devices` (several hot devices) | Out of scope for 2a |

## 5. Synchronization points

| point | who waits for what | existing / new |
|---|---|---|
| Sampling reads logits | host waits for the graph (`llama_context::synchronize`) | existing |
| Start of `process_ubatch` | host: `sched_synchronize` (usually a no-op after sampling); then read the routing log, apply finished promotions and new evictions to the host mirror, write the table tensor once (sync, `cudaStreamPerThread`), push jobs | new (Phase 2a commit point) |
| Before each CPU split at decode | host waits for the CUDA stream (`ggml-backend.cpp:1806`) | existing; untouched in 2a; possible 2b hook |
| Worker: after the staging copy and H2D | worker waits for the transfer stream, then sets "done" | new, off the main thread |
| After the graph | async D2H of the routing log on the compute backend | new, ordered after the graph on the same stream |
| Overlap (if done) | CPU split waits only for the GPU part that produces its inputs | new, scheduler change |

## 6. Required changes per file

### Phase 2a

| file | change | size |
|---|---|---|
| `src/llama-moe-residency.{h,cpp}` (new) | host mirror of the tables; per-layer slot state and LRU; admission; job queue; worker thread with its own `ggml_backend_t` for the device and a pinned staging buffer; commit at the boundary; statistics (hits, misses, promotions, bytes, usefulness, queue depth, drops) | 600-900 lines |
| `src/llama-moe-placement.{h,cpp}` | dynamic mode: every layer both buckets, K_L slots, slot 0 = pinned hottest expert, cold = full merged tensor (mmap required); tables in one contiguous I32 tensor with per-layer views; keep a persistent descriptor (slot tensors, `nb[2]`, merged sources, initial tables) instead of clearing it | 100-200 lines |
| `src/llama-model.{h,cpp}` | hold the descriptor and the routing log tensor (allocated with the placement buffers); graph node count headroom for the extra copy nodes (`n_moe_placement_nodes`) | 30-60 lines |
| `src/llama-graph.cpp` (`build_moe_ffn`) | in dynamic mode, never drop a bucket; `ggml_cpy(ids_flat, view of routing log for layer il)` | 20-40 lines |
| `src/llama-context.{h,cpp}` | own the residency manager (create after the scheduler, free before the backends); `boundary()` call at the start of `process_ubatch`; async read of the routing log after `graph_compute`; statistics in `llama_perf_context_print` | 50-100 lines |
| `include/llama.h` | opt-in params (dynamic on/off, admission window, in-flight limit, byte budget per token); maybe a stats getter | 10-20 lines |
| `common/arg.cpp`, `common/common.{h,cpp}` | `--moe-dynamic` (requires a ranking file), `--moe-admit`, `--moe-promote-budget` | 40-80 lines |
| `src/CMakeLists.txt` | new source file | 1 line |
| `tools/expert-trace/analyze.py` | new section: realistic 2a simulation (warm start from ranking, pinned slot 0, commit delay of 1 token, admission, per-token byte budget, usefulness) | 100-150 lines |
| `tools/expert-trace/compare-logits.py`, `regress-placement.py` | runs with dynamic mode; exactness check with forced promotions (see Risks) | small |
| `ggml/` | none | - |

### Overlap (if done)

| file | change |
|---|---|
| `src/llama-graph.cpp` | graph order: hot FFN after the cold FFN, before the combine |
| `ggml/src/ggml-backend.cpp` | split boundary before a node that reads the output of a split on another backend; launch-ahead of independent async splits before a blocking CPU split; keep the allocator reuse sync (`:1681-1689`) valid |
| `tests/test-backend-ops` or a scheduler test | needs maintainer-style discussion (AGENTS.md); in this fork use the existing `compare-logits.py` exact checks with an RPC device |

## 7. Risks

1. **RAM bandwidth contention.** The staging copy (mmap -> pinned) and the DMA read both take RAM bandwidth that the CPU cold FFN needs; it is measured near its limit. Promotions could slow the misses they are meant to remove. Mitigation: admission, byte budget, measure the CPU time per expert with and without transfers; test `cudaHostRegister` on the mapping to skip the staging copy.
2. **Simulated gain does not carry over.** Commit delay and admission lower the hit rate below 82-89%. Mitigation: simulate first (Recommendation, step 1).
3. **Windows / WDDM.** Async copies may be batched or delayed by the driver; a copy that finishes late only delays promotion, but it may leave the gain small. Verify with Nsight Systems that copies overlap compute.
4. **Thread safety and device selection.** `set_tensor_async` does not call `cudaSetDevice`; the worker must create its backend itself and run on the right device. This is simple with one GPU but must be handled for more.
5. **Testing on this sandbox.** No GPU here. The RPC device can stand in for correctness: table commits and slot copies go through the same ggml calls, and `ggml_backend_dev_init` works for RPC. Speed and concurrency can only be tested on the owner machine.
6. **Correctness of the commit protocol.** A bug that writes a slot still referenced by the tables gives wrong output only sometimes. Mitigation: a debug mode that forces a promotion or eviction on every token with a deterministic policy, and exact logit checks against the unsplit model on CPU/RPC. With the same kernels everywhere, every table state must give bit-exact results.
7. **Memory.** The routing log and staging buffers are small (KiB to a few MiB). The slot count stays as in the Task 4 budget. There is no new large allocation.
8. **Upstream divergence.** 2a is confined to fork-owned files plus small hooks. Overlap edits the scheduler, which changes often upstream, so merges will conflict more.

## 8. Estimated effort (one developer, part time, including verification)

| item | effort |
|---|---|
| Measurement step (trace held-out texts, real static hit rate, timing comparisons) | 0.5-1 day on the owner machine |
| `analyze.py` realistic 2a simulation | 1 day |
| Phase 2a MVP (single GPU, decode, LRU + admission, worker, stats, args) | 1.5-2.5 weeks |
| Phase 2a verification (CPU/RPC exact checks with forced churn; CUDA KLD and speed on the owner machine) | 3-5 days |
| Prefill seeding (after the MVP) | 1-2 days |
| Overlap (scheduler change + checks) | 1-2 weeks, higher risk |
| Phase 2b | open; only after 2a numbers |

## 9. Recommendation: order of work

1. **Measure, no core code (1-2 days).**
   - Trace the held-out prompts (`heldout_code.txt`, the Serbian legal text) with `llama-expert-trace`. Compute the real hit rate of the ranking hot set with `analyze.py` section 4, and LRU section 5 with K matched to the same VRAM.
   - `GGML_SCHED_TIMING=1` for `--n-cpu-moe 48` against the ranking: the GPU time difference is roughly the overlap ceiling.
   - Extend `analyze.py` with the realistic 2a simulation: warm start from the ranking, slot 0 pinned, one-token commit delay, second-miss admission, byte budget from PCIe bandwidth x token time. Report hit rate, bytes per token and usefulness, in domain and out of domain.
2. **Decide with those numbers.** Phase 2a if the realistic simulated hit rate is clearly above the real static one (for example 10 points or more), and decode CPU time is still the largest term. Otherwise overlap first.
3. **Phase 2a MVP** as in section 6: an opt-in `--moe-dynamic` on top of a ranking file, warm start = the Phase 1 hot set, admission off = Phase 1. Verify on CPU/RPC with forced churn (exact), then on CUDA (KLD vs `--cpu-moe`, speed vs the same ranking without `--moe-dynamic`, in and out of domain; the design's section 22 matrix).
4. **Prefill seeding**, then **overlap**, then decide on Phase 2b.

## 10. Not verified in this review

- Everything here comes from reading the code. Nothing was built or run.
- The hot FFN GPU time (2-3 ms) and the real hit rate (about 58%) are derived from the owner's summary numbers and model shapes, not measured directly.
- Whether `cudaHostRegister` on a file mapping works on Windows with the owner's driver.
- Whether async H2D copies overlap compute under WDDM on the RTX 2060 Super.
