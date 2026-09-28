#!/usr/bin/env python3
"""Report on a --moe-dynamic promotion event log (env LLAMA_MOE_DYNAMIC_LOG=<path.csv>).

One row per promotion (see llama_moe_residency::promo_event / write_promo_log in
src/llama-moe-residency.cpp for exactly what each column means and when it is set):

  promo_id, layer, expert, phase (prefill/decode, the ubatch that triggered admission),
  ubatch_size, token_admit, reason (second-miss/force-churn), misses_in_window,
  t_admit_us, t_transfer_start_us, t_transfer_end_us, token_commit,
  demands_while_pending, token_first_reuse, reuse_count, token_evict, evict_reason,
  victim_expert, victim_promo_id (-1: the victim was a warm-start resident, never
  promoted itself), re_misses_after_eviction, bytes.

A promotion is "useful" iff reuse_count > 0 (it was selected by routing at least once
after its transfer committed, before it was itself evicted). This script:

  1. prints hit/reuse/eviction summary stats, split by phase;
  2. prints the reuse-count distribution (0, 1, 2, 3, 4-7, 8+);
  3. estimates net benefit per promotion (reuses * time saved per reuse - transfer
     cost) and reports how many promotions were net positive;
  4. classifies every non-useful promotion into one likely cause, split by phase.

Classification (first match wins, in this order):
  A  threshold too low   - admitted from a prefill ubatch, or on very few misses
                            (misses_in_window <= 1, e.g. --moe-dynamic-force-churn)
  F  slot pressure        - evicted a recent promotion (a short-tenure promoted
                            expert, not a warm-start resident), suggesting the slot
                            was already thrashing before this promotion arrived
  B  evicted too early     - the expert was missed again after this promotion's own
                            eviction (it was still wanted)
  C  transfer too late     - the expert was demanded while the transfer was still
                            pending, but never again after it committed
  E  queue delay           - an unusually long gap between admission and the worker
                            actually starting the transfer (queue/copy backlog)
  D  no longer needed      - no demand at all after admission (the default: none of
                            the above applied)

Example: LLAMA_MOE_DYNAMIC_LOG=promo.csv llama-cli -m model.gguf --moe-dynamic ...
         then: promo-report.py promo.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter, defaultdict


def read_rows(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        int_fields = ["promo_id", "ubatch_size", "token_admit", "misses_in_window", "t_admit_us",
                      "t_transfer_start_us", "t_transfer_end_us", "token_commit", "demands_while_pending",
                      "token_first_reuse", "reuse_count", "token_evict", "victim_expert", "victim_promo_id",
                      "re_misses_after_eviction", "bytes", "layer", "expert"]
        for row in reader:
            for k in int_fields:
                row[k] = int(row[k])
            rows.append(row)
    return rows


def fmt_pct(n, total):
    return f"{n} ({100.0 * n / total:.1f}%)" if total else f"{n} (n/a)"


def print_phase_split(title, rows, pred):
    total = len(rows)
    n = sum(1 for r in rows if pred(r))
    pre = [r for r in rows if r["phase"] == "prefill"]
    dec = [r for r in rows if r["phase"] == "decode"]
    n_pre = sum(1 for r in pre if pred(r))
    n_dec = sum(1 for r in dec if pred(r))
    print(f"  {title}: {fmt_pct(n, total)}  (prefill {fmt_pct(n_pre, len(pre))}, decode {fmt_pct(n_dec, len(dec))})")


def reuse_bucket(n):
    if n <= 3:
        return str(n)
    if n <= 7:
        return "4-7"
    return "8+"


def classify(row, by_id, e_threshold_us):
    if row["phase"] == "prefill" or row["misses_in_window"] <= 1:
        return "A"
    victim = by_id.get(row["victim_promo_id"])
    if victim is not None and victim["token_commit"] >= 0 and victim["token_evict"] >= 0:
        tenure = victim["token_evict"] - victim["token_commit"]
        if tenure < 8:  # ADMIT_WINDOW in src/llama-moe-residency.cpp
            return "F"
    if row["re_misses_after_eviction"] > 0:
        return "B"
    if row["demands_while_pending"] > 0:
        return "C"
    if row["t_transfer_start_us"] > 0 and row["t_admit_us"] > 0:
        wait = row["t_transfer_start_us"] - row["t_admit_us"]
        if wait > e_threshold_us:
            return "E"
    return "D"


CLASS_NAMES = {
    "A": "A threshold too low (prefill / minimal misses)",
    "B": "B evicted too early (re-missed after eviction)",
    "C": "C transfer too late (demanded while pending, never after commit)",
    "D": "D no longer needed (no demand after admission)",
    "E": "E queue delay (large admit-to-start gap)",
    "F": "F slot pressure (victim was a recent promotion)",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="the LLAMA_MOE_DYNAMIC_LOG csv file")
    ap.add_argument("--saved-ms-per-reuse", type=float, default=0.065,
                     help="estimated CPU time saved by one hot (vs cold) expert use, ms (default: %(default)s)")
    ap.add_argument("--transfer-mbs", type=float, default=12000.0,
                     help="assumed transfer bandwidth for the net-benefit estimate, MB/s (default: %(default)s)")
    args = ap.parse_args()

    rows = read_rows(args.log)
    if not rows:
        sys.exit("error: no rows in the log")
    by_id = {r["promo_id"]: r for r in rows}

    total = len(rows)
    useful = [r for r in rows if r["reuse_count"] > 0]
    not_useful = [r for r in rows if r["reuse_count"] == 0]
    pre = [r for r in rows if r["phase"] == "prefill"]
    dec = [r for r in rows if r["phase"] == "decode"]

    print(f"promotions: {total} (prefill {len(pre)}, decode {len(dec)})")
    print_phase_split("useful (reused >= 1 time)", rows, lambda r: r["reuse_count"] > 0)
    print_phase_split("still resident (never evicted)", rows, lambda r: r["token_evict"] < 0)
    print_phase_split("evicted before first commit read back", rows, lambda r: r["token_commit"] < 0)
    total_bytes = sum(r["bytes"] for r in rows)
    print(f"  bytes copied: {total_bytes} ({total_bytes / len(dec) if dec else 0:.0f} / decode-phase promotion)")

    print("\nreuse count distribution:")
    dist = Counter(reuse_bucket(r["reuse_count"]) for r in rows)
    for bucket in ["0", "1", "2", "3", "4-7", "8+"]:
        print(f"  {bucket:>4}: {fmt_pct(dist.get(bucket, 0), total)}")

    print(f"\nnet benefit estimate (saved {args.saved_ms_per_reuse} ms/reuse, transfer @ {args.transfer_mbs} MB/s):")
    net_positive = 0
    net_total_ms = 0.0
    for r in rows:
        transfer_ms = r["bytes"] / (args.transfer_mbs * 1000.0)  # bytes / (MB/s * 1000) = ms
        benefit_ms = r["reuse_count"] * args.saved_ms_per_reuse - transfer_ms
        r["_benefit_ms"] = benefit_ms
        r["_transfer_ms"] = transfer_ms
        net_total_ms += benefit_ms
        if benefit_ms > 0:
            net_positive += 1
    avg_transfer_ms = statistics.mean(r["_transfer_ms"] for r in rows)
    print(f"  avg transfer cost: {avg_transfer_ms:.3f} ms; break-even at {avg_transfer_ms / args.saved_ms_per_reuse:.1f} reuses")
    print(f"  net positive: {fmt_pct(net_positive, total)}")
    print(f"  total net benefit: {net_total_ms:.1f} ms ({'gain' if net_total_ms >= 0 else 'loss'})")

    # E threshold: 2x the median admit-to-transfer-start wait among promotions that did start transferring
    waits = [r["t_transfer_start_us"] - r["t_admit_us"] for r in rows
             if r["t_transfer_start_us"] > 0 and r["t_admit_us"] > 0]
    e_threshold_us = 2 * statistics.median(waits) if waits else float("inf")

    print(f"\nnon-useful promotions: {len(not_useful)} of {total}")
    counts = defaultdict(lambda: [0, 0])  # class -> [prefill, decode]
    for r in not_useful:
        c = classify(r, by_id, e_threshold_us)
        counts[c][0 if r["phase"] == "prefill" else 1] += 1
    for c in ["A", "B", "C", "D", "E", "F"]:
        pre_n, dec_n = counts[c]
        n = pre_n + dec_n
        print(f"  {CLASS_NAMES[c]:<62} {fmt_pct(n, len(not_useful))}  (prefill {pre_n}, decode {dec_n})")


if __name__ == "__main__":
    main()
