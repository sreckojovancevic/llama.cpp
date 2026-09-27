#!/usr/bin/env python3
"""Summarize GGML_SCHED_TIMING=2 output (per-split lines of the ggml scheduler) for decode.

  GGML_SCHED_TIMING=2 llama-completion -m model.gguf -f prompt.txt -n 64 ... 2> log.txt
  sched-timing-summary.py log.txt [--last 48]

Each graph compute ends with a "sched_timing: graph" line. The last N graphs (default: all but the first quarter) are
taken as decode tokens. Prints the average per token by backend kind (CPU / other), and a per-layer table: the layer
of a split is taken from the name of its first node (e.g. "attn_norm-12"). Times are wall times with the split backend
synchronized after each split: "copy" includes waiting for the input and the copy, "compute" the split itself.
"""

from __future__ import annotations

import argparse
import re
import statistics as st
from collections import defaultdict

SPLIT = re.compile(r"sched_timing: split\s+(\d+)\s+(\S+)\s+nodes\s+(\d+)\s+first\s+(\S+)\s+copy\s+([\d.]+) us \(\s*(\d+) B\) compute\s+([\d.]+) us")
GRAPH = re.compile(r"sched_timing: graph ([\d.]+) ms")
LAYER = re.compile(r"-(\d+)(?:\s|$|\()")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--last", type=int, default=0, help="number of graphs at the end to use (default: all but the first quarter)")
    args = ap.parse_args()

    graphs, cur = [], []
    with open(args.log, errors="replace") as f:
        for line in f:
            m = SPLIT.search(line)
            if m:
                cur.append((m.group(2), m.group(4), float(m.group(5)), int(m.group(6)), float(m.group(7))))
                continue
            m = GRAPH.search(line)
            if m:
                graphs.append((float(m.group(1)), cur))
                cur = []
    if not graphs:
        raise SystemExit("no 'sched_timing: graph' lines (run with GGML_SCHED_TIMING=2)")
    n = args.last or max(1, len(graphs) - len(graphs) // 4)
    dec = graphs[-n:]

    kind = lambda be: "CPU" if be.startswith("CPU") else be.split("[")[0]
    tot = defaultdict(lambda: [0, 0.0, 0.0, 0])  # splits, copy us, compute us, bytes (sums over graphs)
    per_layer = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for _, splits in dec:
        layer = -1
        for be, first, cp, b, cm in splits:
            k = kind(be)
            t = tot[k]
            t[0] += 1
            t[1] += cp
            t[2] += cm
            t[3] += b
            # a split without a layer number belongs to the layer of the split before it
            m = LAYER.search(first + " ")
            layer = int(m.group(1)) if m else layer
            per_layer[layer][k][0] += cp
            per_layer[layer][k][1] += cm

    g = len(dec)
    print(f"graphs used: {g} of {len(graphs)}, wall per graph {st.mean(x[0] for x in dec):.2f} ms (median {st.median(x[0] for x in dec):.2f})")
    print(f"{'backend':<10} {'splits':>7} {'compute ms':>11} {'copy+sync ms':>13} {'KiB in':>9}   (average per graph)")
    for k, (ns, cp, cm, b) in sorted(tot.items()):
        print(f"{k:<10} {ns/g:>7.1f} {cm/g/1000:>11.3f} {cp/g/1000:>13.3f} {b/g/1024:>9.1f}")

    kinds = sorted(tot)
    print("\nper layer, average per graph in us (layer -1 = splits without a layer number)")
    print(f"{'layer':>5} " + " ".join(f"{k + ' compute':>14} {k + ' copy':>11}" for k in kinds))
    for layer in sorted(per_layer):
        row = per_layer[layer]
        print(f"{layer:>5} " + " ".join(f"{row[k][1]/g:>14.1f} {row[k][0]/g:>11.1f}" for k in kinds))


if __name__ == "__main__":
    main()
