"""Pre-Stage C detector thresholds (thresholds_v0.json), each at FPR ~0.20 on judged Phase 0 + Stage B claims.
V3 per generator; V5 and V2 pooled. usage: python -m p2.calibrate
"""
import json
import os

from . import db, fusion
from .common import RUNS, jdump

GENS = ["qwen35_122b", "qwen38_27b"]


def main():
    con = db.connect()
    by = {g: [] for g in GENS}
    rows = con.execute("SELECT run_id, gen, block FROM runs WHERE judge_status='done' AND block IN "
                       "('core','big','stageB') AND gen IN (?,?)", GENS).fetchall()
    for r in rows:
        if os.path.exists(os.path.join(RUNS, r["run_id"], "labels.json")):
            by[r["gen"]].append(r["run_id"])
    th = fusion.calibrate(by)
    th["runs"] = {g: len(v) for g, v in by.items()}
    th["source"] = "Phase 0 core/big + Stage B judged runs; Opus 5.5 labels"
    jdump(th, fusion.TH_V0)
    print(json.dumps(th, indent=1))


if __name__ == "__main__":
    main()
