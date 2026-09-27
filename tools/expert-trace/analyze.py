#!/usr/bin/env python3
"""Analyze MoE expert routing traces written by llama-expert-trace.

Sections:
  1. selection histogram per expert
  2. coverage (experts needed for 50/80/95% of selections)
  3. cross-workload stability (Jaccard of top-N hot sets, A vs B)
  4. static placement simulation (VRAM budget, profile A -> hit rate on B)
  5. dynamic cache simulation (per-layer LRU / LFU with K slots)
  6. throughput upper bound for the cold tier
  7. decode tokens/s estimate: LRU experts in VRAM vs all experts on CPU

A and B can each be a comma-separated list of traces; the traces of one group are concatenated.
With one group, each trace is split: first halves of tokens go to A, second halves to B.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import OrderedDict

import numpy as np

TIER_PRESETS = {"nvme": 3000.0, "sata-ssd": 500.0, "hdd": 150.0}  # MB/s


class Trace:
    def __init__(self, name, ubatch, layer, token, rank, expert, meta):
        self.name = name
        self.ubatch = ubatch
        self.layer = layer
        self.token = token
        self.rank = rank
        self.expert = expert
        self.meta = meta

        # decode = ubatch with 1 token
        if len(ubatch):
            pairs = np.unique(np.stack([ubatch, token], axis=1), axis=0)
            ub_ids, ub_ntok = np.unique(pairs[:, 0], return_counts=True)
            self.decode_mask = ub_ntok[np.searchsorted(ub_ids, ubatch)] == 1
        else:
            self.decode_mask = np.zeros(0, dtype=bool)

    @property
    def tokens(self):
        return np.unique(self.token)

    def subset(self, mask, name):
        return Trace(name, self.ubatch[mask], self.layer[mask], self.token[mask], self.rank[mask], self.expert[mask], self.meta)


def concat(traces, name):
    """Join traces one after another; token and ubatch ids are shifted so they do not overlap."""
    if len(traces) == 1:
        t = traces[0]
        return Trace(name, t.ubatch, t.layer, t.token, t.rank, t.expert, t.meta)
    ub, lay, tok, rk, ex = [], [], [], [], []
    ub_off = tok_off = 0
    for t in traces:
        ub.append(t.ubatch + ub_off)
        tok.append(t.token + tok_off)
        lay.append(t.layer)
        rk.append(t.rank)
        ex.append(t.expert)
        ub_off += int(t.ubatch.max()) + 1
        tok_off += int(t.token.max()) + 1
    meta = next((t.meta for t in traces if t.meta), {})
    return Trace(name, *(np.concatenate(x) for x in (ub, lay, tok, rk, ex)), meta)


def split_half(t):
    toks = t.tokens
    cut = toks[len(toks) // 2] if len(toks) > 1 else toks[-1] + 1
    return t.subset(t.token < cut, t.name + " [1st half]"), t.subset(t.token >= cut, t.name + " [2nd half]")


def sidecar_path(csv_path):
    base, ext = os.path.splitext(csv_path)
    if ext == ".csv":
        return base + ".json"
    return csv_path + ".json"


def load_trace(path, meta_path=None):
    meta_path = meta_path or sidecar_path(path)
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        print(f"warning: no metadata file {meta_path}, expert bytes set to 1", file=sys.stderr)

    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 2, 3, 4, 5), dtype=np.int64, ndmin=2)
    if data.size == 0:
        sys.exit(f"error: {path} has no rows")
    t = Trace(os.path.basename(path), data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], meta)
    return t


def parse_bytes(s):
    s = s.strip().upper()
    mult = 1
    for suf, m in (("TB", 1 << 40), ("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10), ("T", 1 << 40), ("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10), ("B", 1)):
        if s.endswith(suf):
            s, mult = s[: -len(suf)], m
            break
    return int(float(s) * mult)


def fmt_bytes(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return str(n)


class Model:
    """Layer list, n_expert and per-layer bytes of one expert."""

    def __init__(self, traces):
        meta = next((t.meta for t in traces if t.meta), {})
        layers = sorted(set(int(x) for t in traces for x in np.unique(t.layer)))
        self.layers = layers
        self.lidx = {l: i for i, l in enumerate(layers)}
        n_exp_trace = max(int(t.expert.max()) for t in traces) + 1
        self.n_expert = max(int(meta.get("n_expert", 0)), n_exp_trace)
        self.n_expert_used = int(meta.get("n_expert_used", 0)) or int(max(t.rank.max() for t in traces)) + 1
        self.model = meta.get("model", "?")
        bytes_by_layer = {int(e["layer"]): int(e["expert_bytes"]) for e in meta.get("layers", [])}
        self.expert_bytes = np.array([bytes_by_layer.get(l, 1) for l in layers], dtype=np.float64)

    def counts(self, t):
        c = np.zeros((len(self.layers), self.n_expert), dtype=np.int64)
        li = np.array([self.lidx[int(l)] for l in t.layer]) if len(t.layer) else np.zeros(0, np.int64)
        np.add.at(c, (li, t.expert), 1)
        return c


def coverage_k(row, frac):
    tot = row.sum()
    if tot == 0:
        return 0
    cs = np.cumsum(np.sort(row)[::-1])
    return int(np.searchsorted(cs, frac * tot) + 1)


def access_stream(m, t):
    """Per layer: list of (token, is_decode, experts) in token order."""
    order = np.lexsort((t.rank, t.token, t.layer))
    streams = {i: [] for i in range(len(m.layers))}
    lay, tok, exp, dec = t.layer[order], t.token[order], t.expert[order], t.decode_mask[order]
    start = 0
    n = len(order)
    while start < n:
        end = start + 1
        while end < n and lay[end] == lay[start] and tok[end] == tok[start]:
            end += 1
        streams[m.lidx[int(lay[start])]].append((int(tok[start]), bool(dec[start]), exp[start:end].tolist()))
        start = end
    return streams


def simulate_cache(streams, k, policy, eb):
    """Return (hits, accesses, miss_bytes_all, miss_bytes_decode)."""
    hits = acc = 0
    mb_all = mb_dec = 0.0
    for li, stream in streams.items():
        if k <= 0:
            for _, dec, exps in stream:
                acc += len(exps)
                mb_all += len(exps) * eb[li]
                if dec:
                    mb_dec += len(exps) * eb[li]
            continue
        cache = OrderedDict()  # expert -> None, order = recency
        freq = {}
        for _, dec, exps in stream:
            for e in exps:
                acc += 1
                freq[e] = freq.get(e, 0) + 1
                if e in cache:
                    hits += 1
                    cache.move_to_end(e)
                    continue
                mb_all += eb[li]
                if dec:
                    mb_dec += eb[li]
                if len(cache) >= k:
                    if policy == "lru":
                        cache.popitem(last=False)
                    else:  # lfu, ties broken by recency
                        victim = min(cache.keys(), key=lambda x: freq[x])
                        del cache[victim]
                cache[e] = None
    return hits, acc, mb_all, mb_dec


def greedy_placement(m, counts_a, budget):
    """Pick (layer, expert) by count/bytes until budget is used. Returns bool mask [n_layer, n_expert] and bytes used.
    Experts not seen in the profile fill the rest of the budget, spread over layers in turn."""
    eb = m.expert_bytes[:, None] * np.ones((1, m.n_expert))
    score = counts_a / eb
    # for unseen experts: n-th unseen expert of each layer comes before the (n+1)-th of any layer
    unseen_rank = np.cumsum(counts_a == 0, axis=1) * (counts_a == 0)
    order = np.lexsort((unseen_rank.ravel(), -score.ravel()))
    placed = np.zeros(counts_a.shape, dtype=bool)
    used = 0.0
    limit = budget * (1 + 1e-9)
    for flat in order:
        li, e = np.unravel_index(flat, counts_a.shape)
        b = eb[li, e]
        if used + b > limit:
            continue
        placed[li, e] = True
        used += b
    return placed, used


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", help="CSV trace groups: first = profile A, second = test B. A group can be comma-separated (a.csv,b.csv) to combine workloads. With one group, A/B = first/second half of each trace.")
    ap.add_argument("--meta", help="JSON sidecar (default: <trace>.json next to the first trace)")
    ap.add_argument("--out-dir", default="expert-trace-report", help="directory for PNG plots (default: %(default)s)")
    ap.add_argument("--top-n", type=int, default=0, help="hot set size per layer for Jaccard (default: n_expert/4)")
    ap.add_argument("--vram-budget", default=None, help="VRAM budget for expert weights, e.g. 8G, 512M, or bytes (default: 50%% of all expert bytes)")
    ap.add_argument("--cache-slots", default=None, help="comma-separated K values for LRU/LFU (default: n_used, 2*n_used, n_expert/4, n_expert/2)")
    ap.add_argument("--bandwidth", default=None, help="comma-separated extra cold-tier bandwidths in MB/s (presets always shown: nvme 3000, sata-ssd 500, hdd 150)")
    ap.add_argument("--pcie-bw", type=float, default=12000.0, help="host to VRAM copy bandwidth in MB/s for section 7 (default: %(default)g)")
    ap.add_argument("--cpu-expert-gbs", type=float, default=40.0, help="effective CPU memory bandwidth for expert compute in GB/s (default: %(default)g)")
    ap.add_argument("--gpu-expert-gbs", type=float, default=400.0, help="effective GPU memory bandwidth for expert compute in GB/s (default: %(default)g)")
    ap.add_argument("--no-plots", action="store_true", help="do not write PNG plots")
    ap.add_argument("--emit-placement", metavar="FILE", default=None, help="write the per-layer hot expert set of section 4 (profile A, --vram-budget) to a JSON file for --moe-placement")
    args = ap.parse_args()

    groups = []
    traces = []
    for g in args.traces:
        grp = []
        for p in g.split(","):
            grp.append(load_trace(p, args.meta if not traces else None))
            traces.append(grp[-1])
        groups.append(grp)
    if len(groups) > 2:
        print(f"note: {len(groups)} trace groups given; only the first two are used as A and B, others only in sections 1-2")
    m = Model(traces)

    if len(groups) == 1:
        halves = [split_half(t) for t in groups[0]]
        A = concat([h[0] for h in halves], "+".join(h[0].name for h in halves))
        B = concat([h[1] for h in halves], "+".join(h[1].name for h in halves))
        ab_note = "single group: A = first half of tokens of each trace, B = second half"
    else:
        A = concat(groups[0], "+".join(t.name for t in groups[0]))
        B = concat(groups[1], "+".join(t.name for t in groups[1]))
        ab_note = f"A = {A.name}, B = {B.name}"

    nL, nE, nU = len(m.layers), m.n_expert, m.n_expert_used
    total_expert_bytes = float(m.expert_bytes.sum() * nE)

    plots = None
    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.out_dir, exist_ok=True)
        plots = plt

    def save(fig, name):
        path = os.path.join(args.out_dir, name)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plots.close(fig)
        print(f"  plot: {path}")

    print(f"model: {m.model}  moe layers: {nL}  n_expert: {nE}  n_expert_used: {nU}")
    print(f"expert bytes per layer (gate+up+down): {fmt_bytes(m.expert_bytes.mean())} avg, all experts: {fmt_bytes(total_expert_bytes)}")
    for t in traces:
        print(f"trace {t.name}: rows {len(t.expert)}, tokens {len(t.tokens)}, decode tokens {len(np.unique(t.token[t.decode_mask]))}")
    print(ab_note)

    # 1. histogram
    print("\n== 1. Selection histogram ==")
    for t in traces:
        c = m.counts(t)
        print(f"-- {t.name}")
        print(f"{'layer':>6} {'top-5 experts (share %)':<52} {'max/mean':>8} {'norm.entropy':>12}")
        for li, l in enumerate(m.layers):
            row = c[li]
            tot = max(row.sum(), 1)
            top = np.argsort(-row)[:5]
            s = " ".join(f"{e}({100.0 * row[e] / tot:.1f})" for e in top)
            p = row[row > 0] / tot
            ent = float(-(p * np.log(p)).sum() / math.log(nE)) if nE > 1 else 1.0
            print(f"{l:>6} {s:<52} {row.max() / (tot / nE):>8.2f} {ent:>12.3f}")
        agg = c.sum(axis=0)
        tot = max(agg.sum(), 1)
        top = np.argsort(-agg)[:5]
        print(f"{'all':>6} {' '.join(f'{e}({100.0 * agg[e] / tot:.1f})' for e in top):<52} {agg.max() / (tot / nE):>8.2f}")
        if plots:
            fig, ax = plots.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [3, 2]})
            share = c / np.maximum(c.sum(axis=1, keepdims=True), 1)
            im = ax[0].imshow(share, aspect="auto", cmap="viridis", interpolation="nearest")
            ax[0].set_xlabel("expert")
            ax[0].set_ylabel("layer index")
            ax[0].set_yticks(range(nL))
            ax[0].set_yticklabels(m.layers)
            ax[0].set_title("selection share per layer")
            fig.colorbar(im, ax=ax[0])
            ax[1].bar(range(nE), np.sort(agg)[::-1] / tot * 100)
            ax[1].axhline(100.0 / nE, color="gray", ls="--", lw=1, label="uniform")
            ax[1].set_xlabel("expert rank (sorted)")
            ax[1].set_ylabel("% of selections (all layers)")
            ax[1].legend()
            fig.suptitle(t.name)
            save(fig, f"1_histogram_{traces.index(t)}.png")

    # 2. coverage
    print("\n== 2. Coverage (experts needed per layer) ==")
    fracs = (0.5, 0.8, 0.95)
    uni = [math.ceil(f * nE) for f in fracs]
    for t in traces:
        c = m.counts(t)
        print(f"-- {t.name}")
        print(f"{'layer':>6} " + " ".join(f"{int(f * 100):>5}%" for f in fracs))
        ks = np.array([[coverage_k(c[li], f) for f in fracs] for li in range(nL)])
        for li, l in enumerate(m.layers):
            print(f"{l:>6} " + " ".join(f"{k:>6}" for k in ks[li]))
        print(f"{'mean':>6} " + " ".join(f"{k:>6.1f}" for k in ks.mean(axis=0)))
        print(f"{'unif.':>6} " + " ".join(f"{k:>6}" for k in uni))
        if plots:
            fig, ax = plots.subplots(figsize=(7, 4.5))
            for li, l in enumerate(m.layers):
                row = np.sort(c[li])[::-1]
                ax.plot(np.arange(1, nE + 1), np.cumsum(row) / max(row.sum(), 1) * 100, lw=1, alpha=0.7)
            ax.plot(np.arange(1, nE + 1), np.arange(1, nE + 1) / nE * 100, "k--", lw=1, label="uniform")
            for f in fracs:
                ax.axhline(f * 100, color="gray", lw=0.5)
            ax.set_xlabel("number of hottest experts")
            ax.set_ylabel("% of selections covered")
            ax.set_title(f"coverage per layer - {t.name}")
            ax.legend()
            save(fig, f"2_coverage_{traces.index(t)}.png")

    cA, cB = m.counts(A), m.counts(B)

    # 3. stability
    top_n = args.top_n if args.top_n > 0 else max(1, nE // 4)
    print(f"\n== 3. Cross-workload stability (Jaccard of top-{top_n} per layer) ==")
    print(ab_note)
    jac = []
    for li, l in enumerate(m.layers):
        sa = set(np.argsort(-cA[li], kind="stable")[:top_n].tolist())
        sb = set(np.argsort(-cB[li], kind="stable")[:top_n].tolist())
        jac.append(len(sa & sb) / len(sa | sb))
    # expected Jaccard of two random top-n sets
    exp_rand = np.mean([len(set(np.random.default_rng(i).choice(nE, top_n, replace=False)) & set(np.random.default_rng(i + 7919).choice(nE, top_n, replace=False))) for i in range(200)])
    exp_rand = exp_rand / (2 * top_n - exp_rand)
    print(f"{'layer':>6} {'jaccard':>8}")
    for l, j in zip(m.layers, jac):
        print(f"{l:>6} {j:>8.3f}")
    print(f"{'mean':>6} {np.mean(jac):>8.3f}   (random sets: ~{exp_rand:.3f})")
    if plots:
        fig, ax = plots.subplots(figsize=(7, 3.5))
        ax.bar(range(nL), jac)
        ax.axhline(exp_rand, color="gray", ls="--", lw=1, label="random")
        ax.set_xticks(range(nL))
        ax.set_xticklabels(m.layers)
        ax.set_ylim(0, 1)
        ax.set_xlabel("layer")
        ax.set_ylabel("Jaccard")
        ax.set_title(f"top-{top_n} overlap, A vs B")
        ax.legend()
        save(fig, "3_stability.png")

    # 4. static placement
    budget = parse_bytes(args.vram_budget) if args.vram_budget else total_expert_bytes * 0.5
    print(f"\n== 4. Static placement (profile A -> test B), budget {fmt_bytes(budget)} ==")
    placed, used = greedy_placement(m, cA, budget)
    totB = max(cB.sum(), 1)
    hitB = float(cB[placed].sum()) / totB
    selfA = float(cA[placed].sum()) / max(cA.sum(), 1)
    n_tok_B = max(len(B.tokens), 1)
    dec_B = B.subset(B.decode_mask, B.name + " decode")
    miss_bytes_B = float(((~placed) * cB * m.expert_bytes[:, None]).sum())
    print(f"placed experts: {int(placed.sum())} / {nL * nE} ({int((placed & (cA == 0)).sum())} not seen in A), bytes used {fmt_bytes(used)} ({100.0 * used / total_expert_bytes:.1f}% of all experts)")
    print(f"hit rate on B: {100.0 * hitB:.2f}%   (on A itself: {100.0 * selfA:.2f}%, uniform guess: {100.0 * used / total_expert_bytes:.2f}%)")
    if len(dec_B.expert):
        cBd = m.counts(dec_B)
        print(f"hit rate on B decode tokens only: {100.0 * cBd[placed].sum() / max(cBd.sum(), 1):.2f}%")
    print(f"missed bytes per token on B: {fmt_bytes(miss_bytes_B / n_tok_B)}")
    print(f"{'layer':>6} {'hot':>5} {'hit % on B':>10}")
    for li, l in enumerate(m.layers):
        print(f"{l:>6} {int(placed[li].sum()):>5} {100.0 * cB[li][placed[li]].sum() / max(cB[li].sum(), 1):>10.2f}")
    if args.emit_placement:
        out = {
            "model": m.model,
            "n_expert": nE,
            "vram_budget": int(budget),
            "bytes_used": int(used),
            "profile": A.name,
            "layers": [{"layer": l, "hot": np.flatnonzero(placed[li]).tolist()} for li, l in enumerate(m.layers)],
        }
        # one line per layer
        body = json.dumps(out, indent=1).split('"layers": [')[0]
        lines = ",\n  ".join(json.dumps(x) for x in out["layers"])
        with open(args.emit_placement, "w") as f:
            f.write(body + '"layers": [\n  ' + lines + "\n ]\n}\n")
        print(f"placement written to {args.emit_placement}")
    sweep = np.linspace(0, 1, 21)
    sweep_hit = []
    for f in sweep:
        p, _ = greedy_placement(m, cA, f * total_expert_bytes)
        sweep_hit.append(float(cB[p].sum()) / totB)
    print("budget sweep (fraction of all expert bytes -> hit rate on B): " + ", ".join(f"{int(f * 100)}%:{100 * h:.1f}" for f, h in zip(sweep[::5], sweep_hit[::5])))
    if plots:
        fig, ax = plots.subplots(figsize=(6, 4.5))
        ax.plot(sweep * 100, np.array(sweep_hit) * 100, "o-", ms=3, label="profile A -> B")
        ax.plot([0, 100], [0, 100], "k--", lw=1, label="uniform")
        ax.axvline(100.0 * budget / total_expert_bytes, color="red", lw=1, label="budget")
        ax.set_xlabel("VRAM budget (% of all expert bytes)")
        ax.set_ylabel("hit rate on B (%)")
        ax.set_title("static placement")
        ax.legend()
        save(fig, "4_static_placement.png")

    # 5. dynamic cache
    if args.cache_slots:
        ks = sorted(set(int(x) for x in args.cache_slots.split(",")))
    else:
        ks = sorted(set(k for k in (nU, 2 * nU, nE // 4, nE // 2) if 0 < k <= nE))
    print(f"\n== 5. Dynamic cache simulation on B (per layer, K slots) ==")
    streams = access_stream(m, B)
    n_dec_B = len(np.unique(B.token[B.decode_mask]))
    print(f"{'K':>5} {'policy':>6} {'hit %':>7} {'bytes/token':>12} {'bytes/decode tok':>16} {'static top-K hit %':>18}")
    cache_res = {}
    for k in ks:
        top = np.argsort(-cA, axis=1, kind="stable")[:, :k]
        st = np.zeros_like(placed)
        np.put_along_axis(st, top, True, axis=1)
        st_hit = 100.0 * cB[st].sum() / totB
        for pol in ("lru", "lfu"):
            h, a, mb, mbd = simulate_cache(streams, k, pol, m.expert_bytes)
            cache_res[(k, pol)] = (h / max(a, 1), mb / n_tok_B, mbd / n_dec_B if n_dec_B else float("nan"))
            dec_s = fmt_bytes(mbd / n_dec_B) if n_dec_B else "n/a"
            print(f"{k:>5} {pol:>6} {100.0 * h / max(a, 1):>7.2f} {fmt_bytes(mb / n_tok_B):>12} {dec_s:>16} {st_hit:>18.2f}")
    print("note: prefill tokens are simulated one token at a time; K < n_expert_used cannot hold one token's experts")
    if plots:
        fig, ax = plots.subplots(figsize=(6, 4.5))
        for pol in ("lru", "lfu"):
            ax.plot(ks, [cache_res[(k, pol)][0] * 100 for k in ks], "o-", label=pol.upper())
        ax.set_xlabel("K slots per layer")
        ax.set_ylabel("hit rate (%)")
        ax.set_title("dynamic cache on B")
        ax.legend()
        save(fig, "5_dynamic_cache.png")

    # 6. throughput bound
    bws = dict(TIER_PRESETS)
    if args.bandwidth:
        for x in args.bandwidth.split(","):
            bws[f"{float(x):g} MB/s"] = float(x)
    print("\n== 6. Throughput upper bound for cold tier (tokens/s = bandwidth / missed bytes per token) ==")
    # use decode tokens when B has them, as decode speed is what the cold tier limits
    cT, n_tok_T = (m.counts(dec_B), n_dec_B) if n_dec_B else (cB, n_tok_B)
    idx = 2 if n_dec_B else 1
    rows = [("static (sec. 4)", float(((~placed) * cT * m.expert_bytes[:, None]).sum()) / n_tok_T)]
    for k in ks:
        for pol in ("lru", "lfu"):
            rows.append((f"{pol.upper()} K={k}", cache_res[(k, pol)][idx]))
    rows.append(("no cache (all cold)", float((cT * m.expert_bytes[:, None]).sum()) / n_tok_T))
    print(f"bytes per token measured on B {'decode tokens' if n_dec_B else 'all tokens (no decode tokens)'}")
    print(f"{'config':<20} {'miss/token':>12} " + " ".join(f"{n:>12}" for n in bws))
    tput = {}
    for name, per in rows:
        vals = [(bw * 1e6) / per if per > 0 else float("inf") for bw in bws.values()]
        tput[name] = vals
        print(f"{name:<20} {fmt_bytes(per):>12} " + " ".join(f"{v:>12.1f}" for v in vals))
    print("note: MB = 1e6 bytes; bound ignores compute and assumes misses are read serially from the cold tier")
    if plots:
        fig, ax = plots.subplots(figsize=(8, 4.5))
        names = list(tput)
        w = 0.8 / len(bws)
        for i, n in enumerate(bws):
            vals = [tput[r][i] for r in names]
            ax.bar(np.arange(len(names)) + i * w, [v if math.isfinite(v) else 0 for v in vals], w, label=n)
        ax.set_xticks(np.arange(len(names)) + 0.4 - w / 2)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_yscale("log")
        ax.set_ylabel("tokens/s upper bound")
        ax.set_title("cold tier throughput bound")
        ax.legend()
        save(fig, "6_throughput_bound.png")

    # 7. decode speed estimate
    pcie = args.pcie_bw * 1e6
    cpu = args.cpu_expert_gbs * 1e9
    gpu = args.gpu_expert_gbs * 1e9
    print("\n== 7. Decode tokens/s estimate: LRU experts in VRAM vs all experts on CPU ==")
    print(f"inputs: pcie {args.pcie_bw:g} MB/s, cpu expert {args.cpu_expert_gbs:g} GB/s, gpu expert {args.gpu_expert_gbs:g} GB/s")
    print("assumptions:")
    print("  - only routed expert FFN time is counted; attention, router, shared experts, dense layers, sync and launch overhead are ignored")
    print("  - expert compute is memory bound: time = expert bytes / effective bandwidth of the device that runs it")
    print("  - one decode token at a time; LRU state as in section 5 (per layer, K slots)")
    print("  - copy:  a miss is copied over PCIe, then computed on GPU; copy and compute do not overlap")
    print("  - cpu:   a miss is computed on CPU from RAM and copied to VRAM in the background for later tokens;")
    print("           GPU hits and CPU misses run one after the other; token time = max(compute time, PCIe copy time)")
    print(f"  - tokens measured: B {'decode tokens' if n_dec_B else 'all tokens (no decode tokens)'}")
    tot_bytes = float((cT * m.expert_bytes[:, None]).sum()) / n_tok_T  # expert bytes used per token
    t_base = tot_bytes / cpu
    print(f"expert bytes read per token: {fmt_bytes(tot_bytes)}")
    print("hit % = share of expert bytes served from VRAM on the measured tokens")
    print(f"{'config':<22} {'VRAM for cache':>14} {'hit %':>7} {'copy tok/s':>11} {'cpu tok/s':>10} {'cpu PCIe %':>10} {'best vs base':>12}")
    print(f"{'all on CPU (base)':<22} {fmt_bytes(0):>14} {'-':>7} {'-':>11} {1 / t_base:>10.1f} {'-':>10} {'1.00x':>12}")
    speed = []
    for k in ks:
        miss = cache_res[(k, "lru")][idx]
        hit = tot_bytes - miss
        t_copy = hit / gpu + miss / pcie + miss / gpu
        t_cmp = hit / gpu + miss / cpu
        t_cpu = max(t_cmp, miss / pcie)
        best = max(1 / t_copy, 1 / t_cpu)
        speed.append((k, 1 / t_copy, 1 / t_cpu))
        vram = float(m.expert_bytes.sum()) * k
        print(f"{f'LRU K={k}':<22} {fmt_bytes(vram):>14} {100.0 * hit / tot_bytes:>7.2f} {1 / t_copy:>11.1f} {1 / t_cpu:>10.1f} {100.0 * (miss / pcie) / t_cpu:>10.1f} {best * t_base:>11.2f}x")
    print(f"{'all in VRAM (ref)':<22} {fmt_bytes(total_expert_bytes):>14} {100.0:>7.2f} {gpu / tot_bytes:>11.1f} {gpu / tot_bytes:>10.1f} {'-':>10} {t_base * gpu / tot_bytes:>11.2f}x")
    print("cpu PCIe % = background copy time as % of token time; at 100% the copies limit the speed")
    if plots:
        fig, ax = plots.subplots(figsize=(6, 4.5))
        ax.plot([s[0] for s in speed], [s[1] for s in speed], "o-", label="miss: copy to VRAM")
        ax.plot([s[0] for s in speed], [s[2] for s in speed], "o-", label="miss: compute on CPU")
        ax.axhline(1 / t_base, color="gray", ls="--", lw=1, label="all on CPU")
        ax.set_xlabel("K slots per layer")
        ax.set_ylabel("decode tokens/s (expert FFN only)")
        ax.set_title("LRU experts in VRAM")
        ax.legend()
        save(fig, "7_decode_estimate.png")


if __name__ == "__main__":
    main()
