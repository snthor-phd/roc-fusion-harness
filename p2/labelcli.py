"""Labeling through the Claude CLI on the Max plan (harness plan v2): extract -> V1 -> judge, all on claude-opus-5-5.

Stage B: labels the 16 new G3 answers and 16 reused Phase 0 G2 answers (copied to stageB__...__G2__capped run ids
so Phase 0's claims and labels stay untouched). Runs sequentially, alongside generation; stops when nothing is left.

usage: python -m p2.labelcli [--block stageB] [--once]
"""
import json
import os
import shutil
import sys
import time
import traceback

from . import db, extract, judge, claude_cli
from .common import RUNS, jload, jdump
from .localq import passages, corpus
from .v1 import run_v1, V1_VERSION

WHO = "labelcli"
TRANSIENT = __import__("re").compile(r"ENOTFOUND|ECONNRESET|ETIMEDOUT|EAI_AGAIN|Can't reach the API|overloaded|rate.?limit|usage limit|limit reached|timed out|Connection (reset|refused)|non-JSON output", __import__("re").I)


def prepare_reuse(con):
    """Copy Phase 0's G2 capped answers for the 8 shared passages into Stage B run ids (generation files only)."""
    from .stageb import design
    n = 0
    for r in design():
        src = {"qwen38_27b": "core__qwen38_27b__%s__G2__capped",
               "qwen35_122b": "big__qwen35_122b__%s__G2__capped"}[r["gen"]] % r["passage"]
        dst = "stageB__%s__%s__G2__capped" % (r["gen"], r["passage"])
        sd, dd = os.path.join(RUNS, src), os.path.join(RUNS, dst)
        if not os.path.exists(os.path.join(sd, "answer.md")):
            db.event(con, WHO, "reuse source missing: %s" % src)
            continue
        if not os.path.exists(dd):
            os.makedirs(dd)
            for f in ("prompt.txt", "answer.md", "thinking.md", "logprobs.json"):
                if os.path.exists(os.path.join(sd, f)):
                    shutil.copy2(os.path.join(sd, f), os.path.join(dd, f))
            m = jload(os.path.join(sd, "meta.json"), {})
            m.update(run_id=dst, block="stageB", reused_from=src, reuse_note="Phase 0 answer, re-extracted and re-judged on Opus 5.5")
            jdump(m, os.path.join(dd, "meta.json"))
            n += 1
        con.execute("""INSERT OR IGNORE INTO runs(run_id, block, gen, passage, lang, grounding, reasoning, repair, rep,
                       seed, ord, parent, gen_status, extract_status, v1_status, judge_status, v5_status, repair_status,
                       v2_status, updated)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'done','pending','pending','pending','pending','na','na',?)""",
                    (dst, "stageB", r["gen"], r["passage"], r["lang"], "G2", "capped", None, 1, None, 100 + r["ord"],
                     src, time.time()))
    con.commit()
    return n


def do_extract(con, r, ps):
    ans = open(os.path.join(RUNS, r["run_id"], "answer.md"), encoding="utf-8").read()
    if not ans.strip():
        extract.save(r["run_id"], {"claims": [], "sections": []}, {"note": "empty answer"})
    else:
        data, meta = claude_cli.run(extract.build_request(r, ps[r["passage"]], ans))
        if isinstance(data.get("claims"), str):
            data["claims"] = json.loads(data["claims"])
        extract.save(r["run_id"], data, meta)
    db.set_status(con, r["run_id"], "extract", "done")


def do_v1(con, r, ps):
    cl = jload(os.path.join(RUNS, r["run_id"], "claims.json"))["claims"]
    jdump(run_v1(corpus(), ps[r["passage"]], cl), os.path.join(RUNS, r["run_id"], "v1.json"))
    jdump({"version": V1_VERSION, "ts": time.time()}, os.path.join(RUNS, r["run_id"], "v1_meta.json"))
    db.set_status(con, r["run_id"], "v1", "done")


def do_judge(con, r, ps):
    rid, p = r["run_id"], ps[r["passage"]]
    cl = jload(os.path.join(RUNS, rid, "claims.json"))["claims"]
    v1 = {x["id"]: x for x in jload(os.path.join(RUNS, rid, "v1.json"))}
    params, key_map, seeds, packet_text, meta = judge.build_request(rid, p, corpus(), cl, v1)
    if meta["n_undecided"] + meta["n_audit"] + meta["n_seeded"] == 0:
        judge.assemble(rid, cl, v1, [], {}, [], "")
    else:
        data, cmeta = claude_cli.run(params)
        labels = data.get("labels")
        if isinstance(labels, str):
            labels = json.loads(labels)
        res = judge.assemble(rid, cl, v1, labels, key_map, seeds, packet_text)
        jdump(dict(cmeta, **{k: meta[k] for k in ("n_undecided", "n_audit", "n_seeded", "rubric")}),
              os.path.join(RUNS, rid, "judge_meta.json"))
        need = {x["id"] for x in res["labels"] if x.get("source") == "missing" and not x.get("seeded")}
        if need:
            sub = [c for c in cl if c["id"] in need]
            sp, skm, _, spt, _ = judge.build_request(rid, p, corpus(), sub, {}, supplement=True)
            sdata, _ = claude_cli.run(sp)
            judge.merge_supplement(rid, sdata.get("labels"), skm, spt, {c["id"]: c for c in cl})
            db.event(con, WHO, "judge supplement %s: %d claims" % (rid, len(sub)))
    db.set_status(con, r["run_id"], "judge", "done")


def step(con, block):
    ps = passages()
    rows = con.execute("SELECT * FROM runs WHERE block=? AND gen_status='done' AND "
                       "(extract_status='pending' OR v1_status='pending' OR judge_status='pending') ORDER BY ord",
                       (block,)).fetchall()
    for r in rows:
        try:
            if r["extract_status"] == "pending":
                do_extract(con, r, ps)
            if con.execute("SELECT v1_status FROM runs WHERE run_id=?", (r["run_id"],)).fetchone()[0] == "pending":
                do_v1(con, r, ps)
            if con.execute("SELECT judge_status FROM runs WHERE run_id=?", (r["run_id"],)).fetchone()[0] == "pending":
                do_judge(con, r, ps)
            db.event(con, WHO, "labeled %s" % r["run_id"])
        except Exception as e:
            if TRANSIENT.search(repr(e)):
                db.event(con, WHO, "label deferred %s (network/service): %s" % (r["run_id"], repr(e)[:200]))
                time.sleep(300)
                return len(rows)
            db.event(con, WHO, "label FAILED %s: %s" % (r["run_id"], traceback.format_exc()[-700:]))
            con.execute("UPDATE runs SET attempts=attempts+1 WHERE run_id=?", (r["run_id"],))
            if (con.execute("SELECT attempts FROM runs WHERE run_id=?", (r["run_id"],)).fetchone()[0] or 0) >= 3:
                for s in ("extract", "v1", "judge"):
                    if r[s + "_status"] == "pending":
                        db.set_status(con, r["run_id"], s, "failed", repr(e)[:300])
    return len(rows)


def main():
    """--block a,b,c (comma list). --until FILE: keep polling until FILE exists and nothing is left (Stage C)."""
    blocks = (sys.argv[sys.argv.index("--block") + 1] if "--block" in sys.argv else "stageB").split(",")
    until = sys.argv[sys.argv.index("--until") + 1] if "--until" in sys.argv else None
    con = db.connect()
    if "stageB" in blocks:
        db.event(con, WHO, "reuse prepared: %d copied" % prepare_reuse(con))
    while True:
        for b in blocks:
            step(con, b)
        q = ",".join("?" * len(blocks))
        left = con.execute("SELECT COUNT(*) FROM runs WHERE block IN (%s) AND gen_status NOT IN ('done','failed')" % q,
                           blocks).fetchone()[0]
        todo = con.execute("SELECT COUNT(*) FROM runs WHERE block IN (%s) AND gen_status='done' AND "
                           "(extract_status='pending' OR v1_status='pending' OR judge_status='pending')" % q,
                           blocks).fetchone()[0]
        if "--once" in sys.argv or (left == 0 and todo == 0 and (until is None or os.path.exists(until))):
            break
        time.sleep(60)
    db.event(con, WHO, "labeling loop finished for %s" % ",".join(blocks))


if __name__ == "__main__":
    main()
