"""Merge per-run d5model.py outputs (D5_ONLY=name) into one JSON."""
import json, sys
out = {"runs": {}, "table": [], "pollution": []}
for p in sys.argv[2:]:
    r = json.load(open(p))
    out["runs"].update(r["runs"])
    out["table"] += r["table"]
    out["pollution"] += r.get("pollution", [])
json.dump(out, open(sys.argv[1], "w"), indent=1)
