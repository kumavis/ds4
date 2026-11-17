#!/usr/bin/env python3
"""D0 gate report: one line per run tag from results.jsonl (latest record).

strict  : div_first_div is null over the whole run, and every teacher-forced
          position's logits are bit-identical (div_bitexact_frac == 1).
quality : the proposed drift budget for non-exact verify settings, from the
          D0 measurements (teacher-forced, per verified position):
            tf_agree       >= 0.97   argmax agrees with plain decode
            flip_margin_max <= 1.0   flips only where plain's top-2 gap < 1 logit
            logit_meanabs  <= 1.5    mean per-position max |logit diff|
Usage: gate_report.py [--results FILE] TAG...
"""
import json, sys

TH = {"tf_agree": 0.97, "flip_margin_max": 1.0, "logit_meanabs": 1.5}


def main(argv):
    path = "/data/ds4-bench/results.jsonl"
    if argv and argv[0] == "--results":
        path, argv = argv[1], argv[2:]
    recs = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        recs[r.get("tag")] = r
    bad = 0
    print(f"{'tag':42} {'setting':8} {'tok/s':>6} {'first_div':>9} {'tf_pos':>6} {'tf_agree':>8} "
          f"{'flips':>5} {'flipmarg':>8} {'maxabs':>7} {'meanabs':>7} {'bitexact':>8}  strict quality")
    for tag in argv:
        r = recs.get(tag)
        if not r:
            print(f"{tag:42} missing")
            bad = 1
            continue
        g = lambda k: r.get("div_" + k)
        strict = r.get("exit_code") == 0 and g("first_div") is None and g("bitexact_frac") == 1
        qual = (r.get("exit_code") == 0 and g("tf_agree") is not None and g("tf_agree") >= TH["tf_agree"]
                and (g("flip_margin_max") or 0) <= TH["flip_margin_max"]
                and (g("logit_meanabs") or 0) <= TH["logit_meanabs"])
        f = lambda v, fmt: "-" if v is None else format(v, fmt)
        print(f"{tag:42} {str(r.get('dspark_verify')):8} {f(r.get('decode_tps'), '6.2f')} "
              f"{f(g('first_div'), '9d')} {f(g('tf_positions'), '6d')} {f(g('tf_agree'), '8.4f')} "
              f"{f(g('tf_flips'), '5d')} {f(g('flip_margin_max'), '8.3f')} {f(g('logit_maxabs'), '7.3f')} "
              f"{f(g('logit_meanabs'), '7.3f')} {f(g('bitexact_frac'), '8.3f')}  "
              f"{'PASS' if strict else 'fail':6} {'PASS' if qual else 'fail'}"
              + (f"  [{r.get('div_error')}]" if r.get("div_error") else ""))
        bad |= not qual
    return bad


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
