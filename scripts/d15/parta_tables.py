#!/usr/bin/env python3
"""D15 part A tables.  Usage: parta_tables.py returns.json pol.json"""
import json
import sys

RET = json.load(open(sys.argv[1]))
POL = json.load(open(sys.argv[2]))
ORDER = ["plain-code", "plain-prose", "plain-json", "plain-translate", "plain-list", "plain-long",
         "plain-code-2k", "plain-prose-2k", "dsp-code", "dsp-prose", "dsp-json", "dsp-translate", "dsp-list"]
print("## Returns\n")
print("| trace | decode misses | returns (share) | reuse tokens p25/p50/p75/p90 | per-layer stack p25/p50/p75/p90 | top-10 layers' share | returning experts | top 10 % experts' share | rho(log next, log prev) | next within 2x prev (random pairing) | P(ret in L+1 given ret in L) (base) | P(partner returns) (base), cases |")
print("|---|---:|---|---|---|---:|---:|---:|---:|---|---|---|")
for t in ORDER:
    o = RET["d12-tr-" + t]
    q, s = o["reuse_tokens_pct"], o["reuse_stack_pct"]
    print(f"| {t} | {o['decode_misses']} | {o['returns']} ({100*o['return_share']:.0f} %) | "
          f"{q['0.25']}/{q['0.5']}/{q['0.75']}/{q['0.9']} | {s['0.25']}/{s['0.5']}/{s['0.75']}/{s['0.9']} | "
          f"{100*o['layer_top10_share']:.0f} % | {o['keys_returning']} | {100*o['key_top10pct_share']:.0f} % | "
          f"{o['spearman_log_next_vs_prev']:.2f} | {100*o['next_within_2x_prev']:.0f} % ({100*o['next_within_2x_random']:.0f} %) | "
          f"{o['p_ret_next_layer_given_ret']:.2f} ({o['p_ret_next_layer_base']:.2f}) | "
          f"{o['p_partner_returns_given_return_and_reselected']:.2f} ({o['p_selected_expert_is_return_miss']:.2f}), {o['partner_cases']} |")
REAL = ["s3fifo", "hawkeye", "hawkeye-sig", "pnu", "pnu-od", "co-same", "co-next", "renewal", "belady"]
BOUND = ["look:0", "look:1", "look:4", "look:16", "look:64", "look:128", "renewal", "renewal-g", "belady"]
for b, pols, title in (("1-128", REAL, "policies"), ("129-512", REAL, "policies"),
                       ("513-2048", REAL, "policies"), ("129-512", BOUND, "bounds")):
    print(f"\n## Tokens {b}, {title}: misses/token under LRU, change of each against LRU\n")
    print("| trace | LRU | " + " | ".join(pols) + " |\n|---|---:|" + "---:|" * len(pols))
    for t in ORDER:
        r = POL["d12-tr-" + t]["policies"]
        if b not in r["lru"]["buckets"]:
            continue
        base = r["lru"]["buckets"][b]["per_token"]
        print(f"| {t} | {base:.1f} | " + " | ".join(
            f"{100*(r[p]['buckets'][b]['per_token']-base)/base:+.0f}%" for p in pols) + " |")
