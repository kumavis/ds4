#!/usr/bin/env bash
# D0 end-to-end gate: plain greedy decode vs DSpark, per model x prompt.
#
#   scripts/d0/gate.sh PREFIX NTOK SETTING [MODE] [models] [prompts]
#     PREFIX   tag prefix, e.g. d0g1
#     SETTING  DSpark verify setting: fast | attn | attn-moe | exact
#     MODE     free (default): free-running DSpark vs plain (first divergence)
#              forced: DSpark teacher-forced along plain's tokens (all positions)
#     models   default "q2 mx"; prompts default "p-code.txt p-prose.txt p-hash.txt"
#
# Each pair is two bench.sh runs (records in results.jsonl): PREFIX-<m>-<p>-plain
# (reused if it already exists) and PREFIX-<m>-<p>-<setting>-<mode> with
# DS4_REF_TAG pointing at the plain run. Then gate_report.py prints the table
# and pass/fail (strict: identical tokens and bit-identical logits; quality:
# the thresholds in gate_report.py). Runs one model process at a time through
# bench.sh (which takes the GPU lock); never interrupt it.
set -u
prefix=$1 ntok=$2 setting=$3 mode=${4:-free}
models=${5:-"q2 mx"} prompts=${6:-"p-code.txt p-prose.txt p-hash.txt"}
here=$(dirname "$(readlink -f "$0")")
dsp="/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-DSpark.gguf"
common=(--nothink --ssd-streaming-cache-experts 82GB)
tags=()
for m in $models; do
  for p in $prompts; do
    pn=${p%.txt}; pn=${pn#p-}
    ref=$prefix-$m-$pn-plain
    if ! grep -q "\"tag\":\"$ref\"" /data/ds4-bench/results.jsonl; then
      DS4_D0_LOGITS=auto DS4_NOTE="d0 gate reference (plain greedy)" \
        "$here/run.sh" "$ref" "$ntok" "$p" "$m" "${common[@]}"
    fi
    tag=$prefix-$m-$pn-$setting-$mode
    force=0; [ "$mode" = forced ] && force=1
    DS4_REF_TAG=$ref DS4_REF_FORCE=$force DS4_DSPARK_VERIFY=$setting \
      DS4_ROCM_DSPARK_SSD_EXPERIMENT=1 DS4_NOTE="d0 gate $setting $mode" \
      "$here/run.sh" "$tag" "$ntok" "$p" "$m" "${common[@]}" --dspark --mtp-model "$dsp"
    tags+=("$tag")
  done
done
nix-shell /data/ds4-env/shell.nix --run "python3 $here/gate_report.py ${tags[*]}" < /dev/null
