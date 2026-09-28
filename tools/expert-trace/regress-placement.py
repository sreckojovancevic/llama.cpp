#!/usr/bin/env python3
"""Regression check for --moe-placement on a tiny random qwen3moe with many layers.

Builds the model (random weights, qwen2 vocab from models/ggml-vocab-qwen2.gguf), quantizes it with llama-quantize,
writes a hot-set file (8 of 16 experts hot per layer) and a ranking file (random counts), and runs compare-logits.py:
both placements must give the same logits as the unsplit model, bit for bit. All runs use -nr, so the repack kernels
do not differ between the runs. Enough layers (default 48) that per-layer costs of the placement show up, e.g. the
graph size limit (8 x n_tensors) that a 4-layer model hides behind its minimum of 1024 nodes.

  regress-placement.py [--build-dir build/bin] [--layers 48] [-- extra llama-debug args, e.g. --rpc 127.0.0.1:50052 -ngl 99]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "gguf-py"))
import gguf  # noqa: E402

N_EMBD, N_HEAD, N_HEAD_KV, HEAD_DIM = 256, 4, 2, 64
N_EXPERT, N_USED, N_FF_EXP = 16, 4, 256


def make_model(path, n_layer):
    vocab = gguf.GGUFReader(os.path.join(ROOT, "models", "ggml-vocab-qwen2.gguf"))
    w = gguf.GGUFWriter(path, "qwen3moe")
    w.add_name("regress-placement")
    w.add_block_count(n_layer)
    w.add_context_length(4096)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF_EXP * N_USED)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_rope_freq_base(1e6)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(N_USED)
    w.add_expert_feed_forward_length(N_FF_EXP)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    for k, f in vocab.fields.items():
        if k.startswith("tokenizer."):
            if f.types[0] == gguf.GGUFValueType.ARRAY:
                w.add_key_value(k, f.contents(), f.types[0], sub_type=f.types[1])
            else:
                w.add_key_value(k, f.contents(), f.types[0])
    n_vocab = len(vocab.fields["tokenizer.ggml.tokens"].contents())

    rng = np.random.default_rng(1234)

    def rand(name, *shape, scale=None):
        s = scale if scale is not None else 1.0 / np.sqrt(shape[-1])
        w.add_tensor(name, (rng.standard_normal(shape) * s).astype(np.float16))

    def ones(name, n):
        w.add_tensor(name, np.ones(n, dtype=np.float32))

    rand("token_embd.weight", n_vocab, N_EMBD, scale=1.0)
    ones("output_norm.weight", N_EMBD)
    for i in range(n_layer):
        p = f"blk.{i}."
        ones(p + "attn_norm.weight", N_EMBD)
        rand(p + "attn_q.weight", N_HEAD * HEAD_DIM, N_EMBD)
        rand(p + "attn_k.weight", N_HEAD_KV * HEAD_DIM, N_EMBD)
        rand(p + "attn_v.weight", N_HEAD_KV * HEAD_DIM, N_EMBD)
        rand(p + "attn_output.weight", N_EMBD, N_HEAD * HEAD_DIM)
        ones(p + "attn_q_norm.weight", HEAD_DIM)
        ones(p + "attn_k_norm.weight", HEAD_DIM)
        ones(p + "ffn_norm.weight", N_EMBD)
        rand(p + "ffn_gate_inp.weight", N_EXPERT, N_EMBD, scale=1.0)
        rand(p + "ffn_gate_exps.weight", N_EXPERT, N_FF_EXP, N_EMBD)
        rand(p + "ffn_up_exps.weight", N_EXPERT, N_FF_EXP, N_EMBD)
        rand(p + "ffn_down_exps.weight", N_EXPERT, N_EMBD, N_FF_EXP)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def make_ranking_dynamic(path, n_layer):
    """A ranking where expert id's count halves with each id (id0 highest), the same for every layer. quantize.py
    can pick a different type per layer for ffn_down_exps (seen: Q4_K on some layers, Q6_K - about 1.46x bigger -
    on others), which changes each layer's count/byte ratio for a tied count; halving every id keeps a >=2x count
    gap between id tiers, comfortably above that, so common_moe_placement_resolve's greedy count/byte fill always
    admits id N of every layer before id N+1 of any layer, regardless of the byte difference. That makes the fill
    a predictable round robin over layers: expert 0 (and then 1) of every layer is always hot first, so every
    layer gets at least 2 hot slots (1 pinned + 1 evictable - expert 0 also ends up pinned at local slot 0,
    create_layer assigns local ids in ascending global-id order) with a small --moe-vram-margin, whatever the
    quant type of that layer, so --moe-dynamic promotions have somewhere to evict, deterministically."""
    counts = [1 << (24 - e) for e in range(N_EXPERT)]
    layers = [{"layer": l, "experts": [{"id": e, "count": counts[e], "bytes": 1} for e in range(N_EXPERT)]} for l in range(n_layer)]
    with open(path, "w") as f:
        json.dump({"profile": "deterministic", "layers": layers}, f)


def pick_dynamic_margin(build_dir, model, ranking_path, extra, target_experts_per_layer, n_layer):
    """Probe the VRAM budget report of the placement device (llama-debug -v, default margin) and derive a
    --moe-vram-margin that leaves room for about target_experts_per_layer experts per layer, on whatever machine
    this runs on. The report line looks like:
      VRAM budget: RPC0: free 16095 MiB - trunk 30 MiB - KV 8 MiB (n_ctx 4096) - compute 301 MiB - margin 512 MiB = 15243 MiB
    budget = free - trunk - kv - compute - margin, so margin = free - trunk - kv - compute - target."""
    exe = ".exe" if os.name == "nt" else ""
    cmd = [os.path.join(build_dir, "llama-debug" + exe), "-m", model, "--moe-placement", ranking_path, "-v",
           "-p", "x", "-n", "0", "-nr"] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout + r.stderr
    m_budget = re.search(r"free (\d+) MiB - trunk (\d+) MiB - KV (\d+) MiB[^-]*- compute (\d+) MiB", out)
    m_hot    = re.search(r"VRAM \(hot\)\s+(\d+)\s+([\d.]+)", out)
    if not m_budget or not m_hot or int(m_hot.group(1)) == 0:
        sys.exit(f"error: could not read a VRAM budget report to pick --moe-dynamic-margin; pass a GPU-like "
                 f"device in the extra args (e.g. --rpc host:port -ngl 99). llama-debug output:\n{out}")
    free_mib, trunk_mib, kv_mib, compute_mib = (int(x) for x in m_budget.groups())
    mib_per_expert = float(m_hot.group(2)) / int(m_hot.group(1))
    target_mib = mib_per_expert * target_experts_per_layer * n_layer
    margin_mib = max(1, int(free_mib - trunk_mib - kv_mib - compute_mib - target_mib))
    return f"{margin_mib}M"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", default=os.path.join(ROOT, "build", "bin"), help="directory with llama-quantize and llama-debug (default: %(default)s)")
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--dynamic", action="store_true",
                     help="also check --moe-dynamic (Phase 2a): needs a GPU-like placement device in the extra "
                          "args (e.g. --rpc host:port -ngl 99), forces churn (LLAMA_MOE_DYNAMIC_FORCE_CHURN) and "
                          "asserts that promotions and evictions actually happened")
    ap.add_argument("--out-dir", default=None, help="keep the model and logs here (default: temp dir)")
    ap.add_argument("extra", nargs="*", help="extra llama-debug args for all runs, after --")
    args = ap.parse_args()

    exe = ".exe" if os.name == "nt" else ""
    out = args.out_dir or tempfile.mkdtemp(prefix="regress-placement-")
    os.makedirs(out, exist_ok=True)
    f16 = os.path.join(out, "model-f16.gguf")
    q = os.path.join(out, "model-q4_k_m.gguf")

    print(f"model: {args.layers} layers, {N_EXPERT} experts, top-{N_USED} -> {q}")
    make_model(f16, args.layers)
    subprocess.run([os.path.join(args.build_dir, "llama-quantize" + exe), f16, q, "Q4_K_M"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(f16)

    rng = np.random.default_rng(7)
    hot = os.path.join(out, "hot.json")
    with open(hot, "w") as f:
        json.dump({"layers": [{"layer": l, "hot": sorted(rng.choice(N_EXPERT, 8, replace=False).tolist())} for l in range(args.layers)]}, f)
    rank = os.path.join(out, "ranking.json")
    with open(rank, "w") as f:
        layers = [{"layer": l, "experts": [{"id": e, "count": int(rng.integers(0, 50)), "bytes": 1} for e in range(N_EXPERT)]} for l in range(args.layers)]
        json.dump({"profile": "random", "layers": layers}, f)

    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "compare-logits.py"),
           "--llama-debug", os.path.join(args.build_dir, "llama-debug" + exe), "-m", q, "--out-dir", os.path.join(out, "logits"),
           "--run", "ref=-nr", "--run", f'hot=--moe-placement "{hot}" -nr', "--run", f'rank=--moe-placement "{rank}" -nr',
           "--exact", "hot", "--exact", "rank", "--"] + args.extra
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)

    if args.dynamic:
        # a separate compare-logits.py call, at --ubatch 1 only: the compute-buffer size for a whole -ub 512
        # ubatch is much larger than this tiny model's total expert weight (a real model wouldn't see this), so
        # one --moe-vram-margin that leaves a small, partial hot/cold split at -ub 1 would leave none of the
        # budget at -ub 512 (all cold) or vice versa. -ub 1 is also the case that matters for --moe-dynamic:
        # PHASE2_DESIGN.md/PHASE2_REVIEW.md scope it to decode, one ubatch boundary per token.
        rank_dyn = os.path.join(out, "ranking-dynamic.json")
        make_ranking_dynamic(rank_dyn, args.layers)
        dyn_extra = args.extra + ["-ub", "1"]
        margin = pick_dynamic_margin(args.build_dir, q, rank_dyn, dyn_extra, target_experts_per_layer=2.5, n_layer=args.layers)
        print(f"moe-dynamic: --moe-vram-margin {margin} (probed)")

        env = os.environ.copy()
        # force every second miss to admit, so a short prompt still exercises promotion and eviction
        env["LLAMA_MOE_DYNAMIC_FORCE_CHURN"] = "1"
        # -lv 5: llama_moe_residency::log_counters() uses LLAMA_LOG_INFO (GGML_LOG_LEVEL_INFO), which
        # common_log_default_callback maps to LOG_LEVEL_TRACE (see common/log.cpp), well above the default
        # verbosity threshold - same as the existing llama_moe_placement::load() summary line, silent by default.
        # also check the admission-policy switches together (each off by default, see WEIGHT_PROVIDER.md): still
        # exact, since they only change which/how many promotions are admitted, never correctness of a promotion
        # once admitted
        switches = "--moe-dynamic-batch-threshold 32 --moe-dynamic-decode-window --moe-dynamic-bw 12000"
        cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "compare-logits.py"),
               "--llama-debug", os.path.join(args.build_dir, "llama-debug" + exe), "-m", q,
               "--out-dir", os.path.join(out, "logits-dynamic"), "--ubatch", "1",
               "--run", "ref=-nr", "--run", f'dyn=--moe-placement "{rank_dyn}" --moe-vram-margin {margin} --moe-dynamic -nr -lv 5',
               "--run", f'dyn_sw=--moe-placement "{rank_dyn}" --moe-vram-margin {margin} --moe-dynamic {switches} -nr -lv 5',
               "--exact", "dyn", "--exact", "dyn_sw", "--"] + dyn_extra
        r = subprocess.run(cmd, env=env)
        if r.returncode != 0:
            sys.exit(r.returncode)

        # exactness alone does not prove the dynamic path was exercised: a no-op residency manager would also
        # pass. Check its counters (log_counters(), printed by llama_moe_residency's destructor at exit) to
        # confirm promotions and evictions (useful promotions) actually happened. A short-lived residency manager
        # is also built for the no_alloc probe pass inside common_moe_placement_resolve (always all-zero, no real
        # inference runs there); take the last (real) occurrence.
        log_dir = os.path.join(out, "logits-dynamic")
        ok = True
        for log in sorted(os.listdir(log_dir)):
            if not ((log.startswith("dyn-ub") or log.startswith("dyn_sw-ub")) and log.endswith(".log")):
                continue
            text = open(os.path.join(log_dir, log)).read()
            matches = list(re.finditer(
                r"moe dynamic residency: hits (\d+) \(prefill (\d+), decode (\d+)\), "
                r"misses (\d+) \(prefill (\d+), decode (\d+)\), promotions (\d+), "
                r"useful promotions (\d+) \(([\d.]+)%\), bytes copied (\d+), decode tokens (\d+), "
                r"bytes/decode token ([\d.]+)", text))
            m = matches[-1] if matches else None
            if not m:
                print(f"error: {log}: no moe dynamic residency counters found")
                ok = False
                continue
            (hits, hits_pre, hits_dec, misses, misses_pre, misses_dec, promotions, useful, useful_pct,
                    bcopied, decode_tokens, bytes_per_dec) = m.groups()
            print(f"{log}: hits {hits} (prefill {hits_pre}, decode {hits_dec}), misses {misses} (prefill "
                  f"{misses_pre}, decode {misses_dec}), promotions {promotions}, useful promotions {useful} "
                  f"({useful_pct}%), bytes copied {bcopied}, decode tokens {decode_tokens}, "
                  f"bytes/decode token {bytes_per_dec}")
            if int(promotions) == 0 or int(useful) == 0:
                print(f"error: {log}: expected promotions > 0 and useful promotions > 0 (churn was forced)")
                ok = False
        if not ok:
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
