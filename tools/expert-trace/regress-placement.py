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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", default=os.path.join(ROOT, "build", "bin"), help="directory with llama-quantize and llama-debug (default: %(default)s)")
    ap.add_argument("--layers", type=int, default=48)
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
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
