"""Stage B (harness plan v2): model screen at G3.

Qwen3.5-122B and Qwen3.8-27B x G3 x the 8 shared design passages (4 Greek, 4 Hebrew), reasoning capped,
one answer per cell = 16 answers. Phase 0's G2 answers for the same cells are reused (re-extracted and
re-judged on Opus 5.5 in the labeling step). Generation yields to Thor via harness.thor_busy().

usage: python -m p2.stageb [--dry]
"""
import random
import sys
import time

from . import db, harness
from .localq import passages, do_gen
from .sampler import SEED

WHO = "stageB"
GENS = ["qwen38_27b", "qwen35_122b"]  # 27B first (faster); each block keeps one model warm in page cache


def design():
    ps = passages()
    half = sorted([p for p in ps.values() if p["split"] == "design" and p.get("half")], key=lambda p: p["id"])
    assert len(half) == 8, len(half)
    rng = random.Random(SEED + 10)
    runs = []
    for gen in GENS:
        block = []
        for p in half:
            block.append({"run_id": "stageB__%s__%s__G3__capped" % (gen, p["id"]), "block": "stageB", "gen": gen,
                          "passage": p["id"], "lang": p["lang"], "grounding": "G3", "reasoning": "capped",
                          "repair": None, "rep": 1})
        rng.shuffle(block)
        runs.extend(block)
    for i, r in enumerate(runs):
        r["ord"] = i
        r["seed"] = rng.randrange(1, 2 ** 31)
    return runs


def register(con, runs):
    for r in runs:
        con.execute("""INSERT OR IGNORE INTO runs(run_id, block, gen, passage, lang, grounding, reasoning, repair, rep,
                       seed, ord, gen_status, extract_status, v1_status, judge_status, v5_status, repair_status,
                       v2_status, updated)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,'pending','pending','pending','pending','pending','na','na',?)""",
                    (r["run_id"], r["block"], r["gen"], r["passage"], r["lang"], r["grounding"], r["reasoning"],
                     r["repair"], r["rep"], r["seed"], r["ord"], time.time()))
    con.commit()


def main():
    runs = design()
    if "--dry" in sys.argv:
        for r in runs:
            print(r["ord"], r["run_id"], r["seed"])
        return
    con = db.connect()
    register(con, runs)
    db.event(con, WHO, "Stage B registered %d runs" % len(runs))
    for r in runs:
        row = con.execute("SELECT gen_status FROM runs WHERE run_id=?", (r["run_id"],)).fetchone()
        if row and row[0] == "done":
            continue
        waited = 0
        while True:
            why = harness.thor_busy()
            if not why:
                break
            if waited % 600 == 0:
                db.event(con, WHO, "yielding before %s: %s" % (r["run_id"], "; ".join(why)))
            time.sleep(60)
            waited += 60
        try:
            do_gen(con, r)
        except Exception as e:
            db.set_status(con, r["run_id"], "gen", "failed")
            db.event(con, WHO, "gen FAILED %s: %r" % (r["run_id"], e))
    harness.kill_llama_servers()
    db.event(con, WHO, "Stage B generation finished")


if __name__ == "__main__":
    main()
