"""Append one correction record to sessions.jsonl (never rewrites existing lines)."""
import datetime as dt, fcntl, json
rec = {
    "schema": 1,
    "kind": "correction",
    "ts": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    "tag": "d17-b5-grow-union-partial",
    "run_dir": "/data/ds4-bench/runs/20261118T170401Z-d17-b5-grow-union-partial",
    "first_invalid_turn": 5,
    "invalid_kind": "turn",
    "faulty_commit": "6aa474fe167c46d9c8703af905b0d75047268769",
    "reason": ("DS4_ROCM_STREAM_STAGING_RECLAIM_UNION (partial staging reclaim) mapped union experts to "
               "lent slots as if the staging were slot-contiguous; it is planar (gate, up, down planes), "
               "so union loads overwrote live cache entries held in lent slots. From turn 5 (the first "
               "union sweep after decode filled the lent slots) the process used corrupted expert weights. "
               "The feature is withdrawn (reverted in d17b-prefill-union). The scenario record of this tag "
               "is also affected. No KV directory was configured; nothing was written to disk caches."),
    "author": "d17",
}
with open("/data/ds4-bench/sessions.jsonl", "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.write(json.dumps(rec, separators=(",", ":")) + "\n")
print("appended")
