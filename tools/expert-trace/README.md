# llama-expert-trace

Records which experts the MoE router selects, per layer and per token, using the eval callback. Core library behavior is not changed.

The tool reads the graph tensor `ffn_moe_topk-<layer>` (set in `build_moe_ffn`, `src/llama-graph.cpp`). It is I32 with shape `[n_expert_used, n_tokens]`.

## Usage

```sh
llama-expert-trace -m model.gguf -p "prompt" -n 128 --trace-out trace.csv
llama-expert-trace -m model.gguf -f prompt.txt -n 256 -ngl 99 --trace-out code.csv
```

Takes the common arguments (`-m`, `-p`, `-f`, `-n`, `-ngl`, `-c`, `-b`, `-ub`, sampling, ...) plus:

- `--trace-out <file>` (required): CSV output. A JSON sidecar is written next to it (`trace.csv` -> `trace.json`).

CSV columns: `ubatch,phase,layer,token,rank,expert`. One row per (layer, token, rank). `phase` is `decode` for an ubatch with 1 token, else `prefill`. `token` is the global token index in the run. `rank` is the position in the top-k list (0 = highest router score).

JSON sidecar: model file name, `arch`, `n_layer`, `n_expert`, `n_expert_used`, `n_layer_moe`, `n_tokens`, `n_ubatch`, `n_rows`, and per MoE layer the bytes of one expert (`gate_bytes`, `up_bytes`, `down_bytes`, `gate_up_bytes`, `expert_bytes` = sum) at the stored quant type. Sizes are read from the GGUF tensor info (all split files are read).

Row count is `n_layer_moe * n_tokens * n_expert_used`.

Notes:

- The prompt is decoded with output for all tokens, in chunks of `n_ubatch`. Otherwise the last layer only runs the output tokens and their routing is lost.
- No warmup run, so the trace has only the real tokens.
- For GroveMoE the traced ids are before the group division.

## Analysis

`analyze.py` needs Python 3 with numpy and matplotlib.

```sh
# one group: A = first half of tokens of each trace, B = second half
python3 tools/expert-trace/analyze.py tools/expert-trace/sample/text.csv

# profile on A, test on B, 4 GiB VRAM budget for expert weights
python3 tools/expert-trace/analyze.py serbian.csv code.csv --vram-budget 4G --out-dir report

# combined profile (comma-separated group) tested on a third workload
python3 tools/expert-trace/analyze.py serbian.csv,code.csv chat.csv

# custom cache sizes and an extra cold-tier bandwidth (MB/s)
python3 tools/expert-trace/analyze.py a.csv b.csv --cache-slots 8,16,32 --bandwidth 7000

# decode estimate with other hardware numbers
python3 tools/expert-trace/analyze.py a.csv b.csv --cache-slots 16,32,64 --pcie-bw 6000 --cpu-expert-gbs 25 --gpu-expert-gbs 350

# write the section 4 hot set (profile A, 6 GiB budget) as a placement file
python3 tools/expert-trace/analyze.py serbian.csv,code.csv chat.csv --vram-budget 6G --emit-placement placement.json
```

The first positional argument is profile A, the second is test B. Each can be a comma-separated list; its traces are joined one after another.

Sections printed to stdout (PNG plots go to `--out-dir`, default `expert-trace-report`):

1. Selection histogram per expert (per layer and all layers).
2. Coverage: experts needed per layer for 50% / 80% / 95% of selections.
3. Stability: Jaccard of top-N hot sets per layer, A vs B (`--top-n`, default `n_expert/4`).
4. Static placement: greedy by count/bytes over all layers from A until `--vram-budget` is used (default 50% of all expert bytes). Experts not seen in A fill the rest of the budget, so a 100% budget gives 100% hits. Hit rate on B, plus a budget sweep.
5. Dynamic cache: per-layer LRU and LFU with K slots on B; hit rate and missed bytes per token and per decode token.
6. Throughput bound: tokens/s = bandwidth / missed bytes per decode token, for NVMe (3000 MB/s), SATA SSD (500), HDD (150) and `--bandwidth` values.
7. Decode estimate: tokens/s for LRU experts in VRAM (K slots per layer) vs all experts on CPU. A miss is either copied over PCIe and computed on GPU, or computed on CPU and copied to VRAM in the background. Only routed expert FFN time is counted, and compute is taken as memory bound. Inputs: `--pcie-bw` (MB/s, default 12000), `--cpu-expert-gbs` (GB/s, default 40), `--gpu-expert-gbs` (GB/s, default 400). The full list of assumptions is printed with the results.

`--emit-placement <file>` writes the hot set of section 4 as JSON, one entry per MoE layer with the sorted global expert ids that go to VRAM:

```json
{
 "model": "model.gguf", "n_expert": 128, "vram_budget": 6442450944, "bytes_used": 6431965184, "profile": "serbian.csv+code.csv",
 "layers": [
  {"layer": 0, "hot": [3, 17, 42]},
  ...
 ]
}
```

With one trace group the profile is only the first half of each trace. To build the placement from whole traces, give a second group (any trace, it is only used as B).

## Sample

`sample/` has two traces (`text.csv`, `code.csv`) from a tiny random-weight llama-arch MoE (6 layers, 16 experts, top-4, Q8_0 experts). They only exercise the tool and the analysis. The routing distribution of a random model says nothing about real models.
