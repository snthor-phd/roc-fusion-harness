"""Judge-matched retention (Phase 1 decision 9): for each of the original answer's true claims (final label S,
attribution excluded), Opus 5.5 decides whether the finished answer still asserts it.

status: kept (same fact, same or near wording) | reworded (same fact, different wording) | dropped (no longer asserted)
| contradicted (finished answer now says otherwise). Retention = (kept + reworded) / all.
usage: python -m p2.retention ORIG_RUN FINAL_RUN [ORIG_RUN FINAL_RUN ...]   -> writes retention.json in each final run
"""
import json
import os
import sys
import time

from . import claude_cli
from .common import RUNS, jload, jdump

SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {
    "id": {"type": "string"}, "status": {"type": "string", "enum": ["kept", "reworded", "dropped", "contradicted"]}},
    "required": ["id", "status"]}}}, "required": ["items"]}

SYSTEM = """You compare an original exegesis answer's verified-true claims with the FINISHED version of that answer (after an automatic repair-and-cut step). For every claim id, decide what the finished answer does with that fact:
- kept: it states the same fact in the same or nearly the same words
- reworded: it states the same fact in different words or in a different place
- dropped: it no longer asserts the fact
- contradicted: it now asserts something incompatible with the fact
Judge only by the finished answer's text. Return one entry for every claim id."""


def claims_for(orig):
    labs = {l["id"]: l for l in jload(os.path.join(RUNS, orig, "labels.json"))["labels"]}
    return [c for c in jload(os.path.join(RUNS, orig, "claims.json"))["claims"]
            if (labs.get(c["id"]) or {}).get("final") == "S" and c.get("type") != "attribution"]


def run_pair(orig, final):
    cl = claims_for(orig)
    ans = open(os.path.join(RUNS, final, "answer.md"), encoding="utf-8").read()
    lines = []
    for c in cl:
        bits = [c["id"], c.get("type") or ""]
        for f in ("ref", "orig", "parse"):
            if c.get(f):
                bits.append("%s: %s" % (f, c[f]))
        bits.append(c.get("claim") or "")
        lines.append(" | ".join(bits))
    user = "CLAIMS (%d):\n%s\n\nFINISHED ANSWER:\n<answer>\n%s\n</answer>" % (len(cl), "\n".join(lines), ans)
    req = {"system": SYSTEM, "messages": [{"role": "user", "content": user}],
           "tools": [{"name": "record_retention", "input_schema": SCHEMA}]}
    data, meta = claude_cli.run(req)
    st = {x["id"]: x["status"] for x in data.get("items", []) if isinstance(x, dict)}
    counts = {k: sum(1 for c in cl if st.get(c["id"]) == k) for k in ("kept", "reworded", "dropped", "contradicted")}
    counts["missing"] = sum(1 for c in cl if c["id"] not in st)
    out = {"orig": orig, "final": final, "n": len(cl), "counts": counts,
           "retention": (counts["kept"] + counts["reworded"]) / len(cl) if cl else None,
           "status": st, "meta": meta, "ts": time.time()}
    jdump(out, os.path.join(RUNS, final, "retention.json"))
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    for o, f in zip(a[0::2], a[1::2]):
        r = run_pair(o, f)
        print(json.dumps({k: r[k] for k in ("final", "n", "counts", "retention")}))
