"""Stage D2 (Thor, 21 Oct): paired rerun of Stage D with budget forcing and lifted caps.

Same frozen pipeline, passages and seeds as Stage D (v1.0.0), except: thinking minimum 16,000 tokens (budget forcing,
up to 6 "Wait," extensions), thinking cap 32,768, answer cap 32,768, server context 131,072 (Qwen's guidance), applied
to both the draft and the repair pass. Same thresholds (v1). Decision rule: adopt budget forcing only if the paired
per-passage delivered-HR difference (D2 - D) has a 95% CI entirely below 0; otherwise keep v1.0.0 with the lifted caps.
usage: python -m p2.stageD2
"""
import os

os.environ.setdefault("P2_MIN_THINK", "16000")
os.environ.setdefault("P2_THINK_CAP", "32768")
os.environ.setdefault("P2_ANSWER_MAX", "32768")
os.environ.setdefault("P2_CTX", "131072")
os.environ.setdefault("P2_MAX_FORCES", "6")

from . import stagec, db, harness  # noqa: E402
from .common import STATE  # noqa: E402

stagec.CFG.update({"prefix": "stageD2", "split": "evaluation", "n_passages": 24, "webs": (0,), "v2_n": 0,
                   "cells": [(0, "2of3")], "th_path": os.path.join(STATE, "thresholds_v1.json"), "th_label": "v1",
                   "block": "stageD2", "dblock": "stageD2_delivered", "fblock": "stageD2_final",
                   "done": "STAGED2_DONE", "ord_base": 3000})  # ord_base 3000 = Stage D's, so seeds match
stagec.WHO = "stageD2"


def main():
    con = db.connect()
    db.event(con, "stageD2", "start: min_think %s, think_cap %s, answer_max %s, ctx %s" % (
        harness.MIN_THINK, harness.THINK_CAP, harness.ANSWER_MAX, stagec.CTX[1]))
    pool = stagec.Server()
    try:
        stagec.run_gen("qwen35_122b", con, pool)
    finally:
        pool.stop()
        harness.kill_llama_servers()
    open(os.path.join(STATE, "STAGED2_DONE"), "w").write(__import__("time").ctime())
    db.event(con, "stageD2", "Stage D2 generation and cuts finished")


if __name__ == "__main__":
    main()
