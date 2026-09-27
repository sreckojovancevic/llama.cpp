#!/usr/bin/env python3
"""Compare last-token logits of llama-debug runs, e.g. --moe-placement against the unsplit model.

Each run is llama-debug with the common args plus the run's own args, at each --ubatch size.
The first run is the reference; the others are compared to it at the same ubatch size.

  --exact NAME   the run must match the reference exactly (max abs diff 0)
  --tol X        other runs must have the same top-1 token and max abs diff <= X (default: only report)

Use it for exact checks: two runs that compute every op with the same kernels must give the same bits.
Runs that use other kernels (e.g. some experts on the GPU instead of the CPU) differ a little, and on a
single prompt that small difference can flip a near-tied router choice and change the logits a lot, so do
not judge those by one prompt: use llama-perplexity --kl-divergence over many tokens instead.

Example (CUDA): all-cold placement vs --cpu-moe must be exact
  compare-logits.py --llama-debug build/bin/llama-debug -m model.gguf \\
      --run "ref=--cpu-moe" --run "cold=--moe-placement all-cold.json" --exact cold -- -ngl 99
"""

from __future__ import annotations

import argparse
import glob
import os
import shlex
import subprocess
import sys
import tempfile

import numpy as np

DEFAULT_PROMPT = (
    "The history of the city goes back more than two thousand years. It was founded on a hill above the river, "
    "where two old trade roads met. def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n"
    "Danas je grad poznat po mostovima, pijacama i kafanama. The quick brown fox jumps over the lazy dog."
)


def run_debug(args, name, run_args, ubatch, out_root):
    out_dir = os.path.join(out_root, f"{name}-ub{ubatch}")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [args.llama_debug, "-m", args.model, "-p", args.prompt, "--save-logits", "--logits-output-dir", out_dir,
           "-b", "2048", "-ub", str(ubatch)] + shlex.split(run_args) + args.common
    log = os.path.join(out_root, f"{name}-ub{ubatch}.log")
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        sys.exit(f"error: run '{name}' failed (exit {r.returncode}), see {log}\n  {shlex.join(cmd)}")
    logits = [p for p in glob.glob(os.path.join(out_dir, "*.bin")) if not p.endswith("-tokens.bin")]
    tokens = glob.glob(os.path.join(out_dir, "*-tokens.bin"))
    if len(logits) != 1 or len(tokens) != 1:
        sys.exit(f"error: run '{name}' did not write logits to {out_dir}")
    return np.fromfile(logits[0], dtype=np.float32), np.fromfile(tokens[0], dtype=np.int32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-debug", default="build/bin/llama-debug", help="path to llama-debug (default: %(default)s)")
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--run", action="append", metavar="NAME=ARGS", required=True, help="named run with extra llama-debug args, e.g. \"ref=--cpu-moe\"; the first run is the reference")
    ap.add_argument("--exact", action="append", default=[], metavar="NAME", help="run that must match the reference exactly")
    ap.add_argument("--tol", type=float, default=None, help="allowed max abs diff for the runs not in --exact (default: only report)")
    ap.add_argument("--ubatch", default="512,1", help="comma-separated ubatch sizes (default: %(default)s; 1 = one token per graph, like decode)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--out-dir", default=None, help="keep logits and logs here (default: temp dir)")
    ap.add_argument("common", nargs="*", help="args for all runs, after --")
    args = ap.parse_args()
    if any("=" not in r for r in args.run):
        sys.exit("error: --run takes NAME=ARGS")
    args.run = [r.split("=", 1) for r in args.run]

    names = [r[0] for r in args.run]
    for n in args.exact:
        if n not in names[1:]:
            sys.exit(f"error: '{n}' is not a run name (the reference cannot be used)")

    out_root = args.out_dir or tempfile.mkdtemp(prefix="compare-logits-")
    ok = True
    print(f"reference: {names[0]} ({args.run[0][1]}), common args: {shlex.join(args.common)}, logs: {out_root}")
    for ub in [int(x) for x in args.ubatch.split(",")]:
        res = {name: run_debug(args, name, run_args, ub, out_root) for name, run_args in args.run}
        ref, ref_tok = res[names[0]]
        top_ref = np.argsort(-ref)[:10]
        print(f"\n-- ubatch {ub}, {len(ref_tok)} prompt tokens, max |logit| {np.abs(ref).max():.3g}")
        print(f"{'run':<12} {'max abs diff':>12} {'rel':>10} {'top-1':>6} {'top-10 overlap':>14}  check")
        for name in names[1:]:
            lg, tok = res[name]
            if not np.array_equal(tok, ref_tok) or lg.size != ref.size:
                sys.exit(f"error: run '{name}' used other tokens than the reference")
            d = float(np.abs(lg - ref).max())
            top = np.argsort(-lg)[:10]
            same1 = bool(top[0] == top_ref[0])
            ov = len(set(top.tolist()) & set(top_ref.tolist()))
            if name in args.exact:
                good, rule = d == 0.0, "exact"
            elif args.tol is not None:
                good, rule = same1 and d <= args.tol, f"top-1 and diff <= {args.tol:.3g}"
            else:
                good, rule = True, "report only"
            ok &= good
            print(f"{name:<12} {d:>12.4g} {d / max(np.abs(ref).max(), 1e-30):>10.3g} {str(same1):>6} {ov:>14}  {'OK' if good else 'FAIL'} ({rule})")
    print("\nresult:", "OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
