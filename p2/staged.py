"""Stage D: the frozen harness (harness.lock v1.0.0) on the unseen evaluation split.

Qwen3.5-122B, G3, reasoning capped, web off, V2 off, fusion "2 of 3" over V3 + V5 (both must flag) at
thresholds_v1, one repair pass on the own packet, re-check, cut. 24 passages (12 Greek, 12 Hebrew): 24 drafts and
24 finished answers, both judged; judge-matched retention on every pair.
usage: python -m p2.staged
"""
import os
import subprocess
import sys

from . import stagec, db, harness
from .common import STATE, REPO

stagec.CFG.update({"prefix": "stageD", "split": "evaluation", "n_passages": 24, "webs": (0,), "v2_n": 0,
                   "cells": [(0, "2of3")], "th_path": os.path.join(STATE, "thresholds_v1.json"), "th_label": "v1",
                   "block": "stageD", "dblock": "stageD_delivered", "fblock": "stageD_final", "done": "STAGED_DONE",
                   "ord_base": 3000})
stagec.WHO = "stageD"


def main():
    r = subprocess.run([sys.executable, "-m", "p2.lock", "verify"], cwd=REPO, capture_output=True, text=True)
    con = db.connect()
    if r.returncode != 0:
        db.event(con, "stageD", "lock verify FAILED: %s" % (r.stdout + r.stderr)[-400:])
        raise SystemExit("harness.lock mismatch: " + r.stdout)
    db.event(con, "stageD", "lock verified: %s" % r.stdout.strip())
    pool = stagec.Server()
    try:
        stagec.run_gen("qwen35_122b", con, pool)
    finally:
        pool.stop()
        harness.kill_llama_servers()
    db.event(con, "stageD", "Stage D generation and cuts finished")
    open(os.path.join(STATE, "STAGED_DONE"), "w").write(__import__("time").ctime())


if __name__ == "__main__":
    main()
