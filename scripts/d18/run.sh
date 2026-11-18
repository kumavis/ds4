#!/usr/bin/env bash
# D18 copy of D14 run.sh (tags d18-*, driver from scripts/d14). D14 session benchmark: one bounded ds4-server session driven by driver.py.
#
# Usage: run.sh TAG SCENARIO CTX "SERVER_EXTRA_FLAGS" STEP [STEP ...]
#
# Env:
#   DS4_DIR        ds4 checkout whose ds4-server runs (default /data/ds4-int2; not modified)
#   DS4_MODEL      model path (default MXFP4 1017)
#   D14_PORT       server port on 127.0.0.1 (default 18714)
#   D14_PAYLOAD    payload JSON (default /data/ds4-bench/d14-payload.json)
#   D14_BUDGET_MIN stop adding turns after this many minutes of server time (default 38)
#   DS4_NOTE       free text stored in the records
#   DS4_* / HSA_*  passed to ds4-server and recorded
#
# Output: /data/ds4-bench/runs/<UTC>-<TAG>/{run.log,server.log,sys.log,turns.jsonl,
# transcript-*.json,driver.json}, records appended to /data/ds4-bench/sessions.jsonl.
#
# Rules: holds /data/ds4-bench/gpu.lock for the server's whole life (the server
# inherits the lock fd). The server is stopped only by driver.py's clean stop:
# one SIGTERM once no request is in flight, then wait. The unit is created with
# SendSIGKILL=no and TimeoutStopSec=infinity. Never kill this unit.
set -u
if [ $# -lt 5 ]; then sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 2; fi
BENCH_ROOT=/data/ds4-bench
HERE=/data/ds4-d14/scripts/d14   # driver.py from D14, unchanged
tag=$1 scenario=$2 ctx=$3 extra=$4; shift 4
case $tag in d18-*) ;; *) echo "run: TAG must start with d18-" >&2; exit 2;; esac
case $tag in *[!A-Za-z0-9._-]*) echo "run: bad TAG" >&2; exit 2;; esac

if [ -z "${INVOCATION_ID:-}" ]; then
  unit="d18-$tag-$(date -u +%Y%m%dT%H%M%SZ)"
  envargs=(-E "PATH=$PATH" -E "HOME=$HOME")
  while IFS= read -r v; do envargs+=(-E "$v=${!v}"); done < <(compgen -e | grep -E '^(DS4_|HSA_|D14_)')
  echo "run: launching user unit $unit (journalctl --user -u $unit)"
  exec systemd-run --user --collect --quiet --unit="$unit" -p SendSIGKILL=no -p TimeoutStopSec=infinity \
    -p KillMode=mixed --working-directory=/data/ds4-bench "${envargs[@]}" \
    "$(readlink -f "$0")" "$tag" "$scenario" "$ctx" "$extra" "$@"
fi

DS4_DIR=$(readlink -f "${DS4_DIR:-/data/ds4-int2}")
model=${DS4_MODEL:-/data/ds4/gguf/DeepSeek-V4.1-Flash-1017-MXFP4-experts.gguf}
port=${D14_PORT:-18714}
payload=${D14_PAYLOAD:-$BENCH_ROOT/d14-payload.json}
budget=${D14_BUDGET_MIN:-38}
export DS4_ROCM_STREAM_CACHE_STATS=${DS4_ROCM_STREAM_CACHE_STATS:-1}
export DS4_ROCM_STREAM_READ_PROFILE=${DS4_ROCM_STREAM_READ_PROFILE:-1}
export DS4_DSPARK_STATS=${DS4_DSPARK_STATS:-1}
export DS4_D8_STEP_TRACE=${DS4_D8_STEP_TRACE:-1}
export DS4_DSPARK_SPEC_LOG=${DS4_DSPARK_SPEC_LOG:-1}
read -r -a xflags <<< "$extra"
# greedy and non-thinking are per request (temperature 0, think false): the
# server has no --temp/--nothink flags
flags=(--rocm --ssd-streaming --ctx "$ctx" --host 127.0.0.1 --port "$port" "${xflags[@]}")

exec 9>>"$BENCH_ROOT/gpu.lock"
if ! flock -n 9; then echo "run: waiting for gpu.lock"; flock 9; fi
if pgrep -x 'ds4|ds4-server|ds4-bench|ds4-eval|ds4-agent' >/dev/null; then
  echo "run: a ds4 process is already running; refusing:" >&2; pgrep -a -x 'ds4|ds4-server|ds4-bench|ds4-eval|ds4-agent' >&2; exit 3
fi
if (exec 3<>/dev/tcp/127.0.0.1/$port) 2>/dev/null; then echo "run: port $port busy" >&2; exit 3; fi

nvme_max_c() { local m=- f t; for f in /sys/class/nvme/nvme*/hwmon*/temp1_input; do [ -r "$f" ] || continue
  t=$(( $(cat "$f") / 1000 )); if [ "$m" = - ] || [ "$t" -gt "$m" ]; then m=$t; fi; done; echo "$m"; }
cool_start=$(date +%s)
while [ "$(nvme_max_c)" != - ] && [ "$(nvme_max_c)" -gt "${DS4_SSD_COOL_C:-72}" ] && [ $(( $(date +%s) - cool_start )) -lt 300 ]; do sleep 5; done

start=$(date +%s)
out=$BENCH_ROOT/runs/$(date -u -d "@$start" +%Y%m%dT%H%M%SZ)-$tag
mkdir -p "$out"
commit=$(git -C "$DS4_DIR" rev-parse HEAD 2>/dev/null || echo "")
if [ -n "$(git -C "$DS4_DIR" status --porcelain --untracked-files=no 2>/dev/null)" ]; then dirty=true; else dirty=false; fi

dev=$(basename "$(readlink -f /dev/mapper/data)")
card=$(dirname "$(ls /sys/class/drm/card*/device/gpu_busy_percent | head -1)")
(
  rd() { awk '$3 ~ /^nvme[0-9]+n[0-9]+$/ {s+=$6} END{print s}' /proc/diskstats; }
  read -r prev_d < <(awk -v d="$dev" '$3==d{print $6}' /proc/diskstats); prev_n=$(rd)
  while true; do
    sleep 2
    read -r cur_d < <(awk -v d="$dev" '$3==d{print $6}' /proc/diskstats); cur_n=$(rd)
    echo "$(date +%T) epoch=$(date +%s) read_mbs=$(( (cur_d - prev_d) * 512 / 2 / 1048576 )) nvme_read_mbs=$(( (cur_n - prev_n) * 512 / 2 / 1048576 )) gpu_busy=$(cat "$card/gpu_busy_percent") gtt_mb=$(( $(cat "$card/mem_info_gtt_used") / 1048576 )) $(free -m | awk '/Mem:/{print "avail_mb="$7" cache_mb="$6}') load=$(cut -d' ' -f1 /proc/loadavg) nvme_max_c=$(nvme_max_c)"
    prev_d=$cur_d prev_n=$cur_n
  done
) > "$out/sys.log" 2>&1 9>&- &
sampler=$!

cmd=(./ds4-server -m "$model" "${flags[@]}")
{
  echo "run: tag=$tag scenario=$scenario start=$(date -d "@$start" -Iseconds) ds4_dir=$DS4_DIR commit=$commit dirty=$dirty"
  echo "run: model=$(readlink -f "$model") payload=$payload budget_min=$budget"
  echo "run: nvme_max_start_c=$(nvme_max_c) cooldown_wait_s=$(( start - cool_start ))"
  echo "run: cmd=${cmd[*]}"
  echo "run: steps=$*"
  echo "run: env=$(env | grep -E '^(DS4_|HSA_)' | sort | paste -sd' ' -)"
} > "$out/run.log"
echo "run: $tag -> $out"

cd "$DS4_DIR"
printf -v qcmd '%q ' "${cmd[@]}"
nix-shell /data/ds4-env/shell.nix --run "echo \$\$ > $out/server.pid; exec $qcmd" > "$out/server.log" 2>&1 < /dev/null &
srv=$!

stepargs=(); for s in "$@"; do stepargs+=(--step "$s"); done
nix-shell /data/ds4-env/shell.nix --run "$(printf '%q ' python3 "$HERE/driver.py" run --run-dir "$out" --port "$port" \
  --payload "$payload" --ctx "$ctx" --budget-min "$budget" "${stepargs[@]}")" >> "$out/run.log" 2>&1 < /dev/null 9>&-
drc=$?
# The driver stops the server cleanly at its end. If it failed to, try the same
# clean stop once more (it refuses unless the server is idle); never kill.
if kill -0 "$srv" 2>/dev/null; then
  nix-shell /data/ds4-env/shell.nix --run "python3 $HERE/driver.py stop --run-dir $out --port $port" >> "$out/run.log" 2>&1 < /dev/null 9>&-
fi
wait "$srv"; rc=$?
end=$(date +%s)
echo "EXIT=$rc wall_s=$(( end - start )) driver_rc=$drc" >> "$out/run.log"
kill "$sampler" 2>/dev/null
exec 9>&-
nix-shell /data/ds4-env/shell.nix --run "$(printf '%q ' python3 "$HERE/driver.py" finalize --run-dir "$out" --tag "$tag" \
  --scenario "$scenario" --note "${DS4_NOTE:-}" --ctx "$ctx" --model "$model" --payload "$payload" --ds4-dir "$DS4_DIR" \
  --ds4-commit "$commit" --ds4-dirty "$dirty" --start "$start" --end "$end" -- "${flags[@]}")" >> "$out/run.log" 2>&1 < /dev/null
echo "run: done rc=$rc" >> "$out/run.log"
exit "$rc"
