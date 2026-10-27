"""Stage C (harness plan v2): harness tune on the 8 calibration passages, run for BOTH generators (Thor, 19 Oct 22:54).

Factors (2^3, split-plot): web research {off, on} (whole plot: changes the generated answer) x resampling detector V2
{off, on} x fusion rule {any, 2of3} (sub-plots: change only the flags). Grounding G3, reasoning capped at 8,000.
Per generator: 16 originals (8 passages x web), 80 V2 resamples, 64 delivered answers (16 x V2 x rule), each
repaired once on its own packet, re-checked, flagged claims cut -> 64 final answers judged. Originals judged too.

Phases are idempotent (each checks its output files) so the run can be stopped and restarted.
usage: python -m p2.stagec [--gens qwen35_122b,qwen38_27b] [--dry]
"""
import os
import random
import re
import shutil
import sys
import time
import traceback

from . import db, harness, fusion, web
from .common import RUNS, STATE, jload, jdump
from .localq import passages, corpus, write
from .prompts import base_prompt, generation_prompt
from .repair import REPAIR_TMPL, own_packet
from .sampler import SEED

WHO = "stageC"
GENS = ["qwen35_122b", "qwen38_27b"]
V2_N = 5
RULES = ["any", "2of3"]
CTX = ["-c", os.environ.get("P2_CTX", "65536")]  # repair prompts carry the whole previous answer
# Stage C settings; Stage D (p2.staged) overwrites CFG for the frozen harness on the evaluation split.
CFG = {"prefix": "stageC", "split": "calibration", "n_passages": 8, "webs": (0, 1), "v2_n": V2_N,
       "cells": [(v, r) for v in (0, 1) for r in RULES], "th_path": None, "th_label": "v0",
       "block": "stageC", "dblock": "stageC_delivered", "fblock": "stageC_final", "done": "STAGEC_DONE",
       "ord_base": 1000}
WEB_NOTE = ("\n\nWhen you use a fact from the WEB RESEARCH notes, cite its tag, e.g. [X:2]. Do not cite web pages "
            "that are not in the notes.")


# ------------------------------------------------------------------ design
def originals(gen):
    ps = passages()
    cal = sorted([p for p in ps.values() if p["split"] == CFG["split"]], key=lambda p: p["id"])
    assert len(cal) == CFG["n_passages"], len(cal)
    rng = random.Random(SEED + 20 + GENS.index(gen))
    out = []
    for p in cal:
        for w in CFG["webs"]:
            out.append({"run_id": "%s__%s__%s__G3__web%d" % (CFG["prefix"], gen, p["id"], w), "block": CFG["block"], "gen": gen,
                        "passage": p["id"], "lang": p["lang"], "grounding": "G3", "reasoning": "capped",
                        "repair": None, "rep": 1, "web": w})
    rng.shuffle(out)
    for i, r in enumerate(out):
        r["ord"], r["seed"] = CFG["ord_base"] * (GENS.index(gen) + 1) + i, rng.randrange(1, 2 ** 31)
    return out


def resamples(o):
    rng = random.Random(hash((SEED, o["run_id"])) & 0xFFFFFFF)
    return [dict(o, run_id=o["run_id"] + "__v2s%d" % k, block="v2gen", parent=o["run_id"], rep=k + 1,
                 seed=rng.randrange(1, 2 ** 31), k=k) for k in range(1, CFG["v2_n"] + 1)]


def cells(o):
    return [dict(o, v2=v, rule=rule, run_id="%s__v2%d__%s" % (o["run_id"], v, rule)) for v, rule in CFG["cells"]]


def register(con, r, gen_status="pending", extract="pending", judge="pending", parent=None):
    con.execute("""INSERT OR IGNORE INTO runs(run_id, block, gen, passage, lang, grounding, reasoning, repair, rep, seed,
                   ord, parent, gen_status, extract_status, v1_status, judge_status, v5_status, repair_status,
                   v2_status, updated) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'pending','na','na',?)""",
                (r["run_id"], r["block"], r["gen"], r["passage"], r["lang"], r["grounding"], r["reasoning"],
                 r.get("repair"), r.get("rep", 1), r.get("seed"), r.get("ord", 0), parent or r.get("parent"),
                 gen_status, extract, "pending" if extract == "pending" else extract, judge, time.time()))
    con.commit()


def status(con, rid, col):
    row = con.execute("SELECT %s FROM runs WHERE run_id=?" % col, (rid,)).fetchone()
    return row[0] if row else None


# ------------------------------------------------------------------ helpers
def wait_for_thor(con, what):
    waited = 0
    while True:
        why = harness.thor_busy()
        if not why:
            return
        if waited % 1800 == 0:
            db.event(con, WHO, "yielding before %s: %s" % (what, "; ".join(why)))
        time.sleep(60)
        waited += 60


class Server:
    """Keeps one llama-server loaded across many calls."""

    def __init__(self):
        self.key, self.srv = None, None

    def get(self, gen):
        if self.key != gen:
            self.stop()
            self.srv = harness.LlamaServer(gen, extra_args=CTX)
            self.srv.start()
            self.key = gen
        return self.srv

    def stop(self):
        if self.srv:
            self.srv.stop()
        self.key, self.srv = None, None


def gen_files(con, srv, r, prompt, out_id=None, extra_meta=None):
    rid = out_id or r["run_id"]
    d = os.path.join(RUNS, rid)
    os.makedirs(d, exist_ok=True)
    write(os.path.join(d, "prompt.txt"), prompt)
    cfg = harness.GENERATORS[r["gen"]]
    meta = {"run_id": rid, "block": r["block"], "gen": r["gen"], "model_label": cfg["label"], "stack": cfg["stack"],
            "passage": r["passage"], "lang": r["lang"], "grounding": r["grounding"], "reasoning": r["reasoning"],
            "engine": "llama-server " + harness.llama_version(), "weights": os.path.basename(cfg["gguf"]),
            "server_args": cfg["args"] + CTX, "started": time.time(), "host": os.uname().nodename}
    meta.update(extra_meta or {})
    answer, thinking, lps, m = harness.generate_llama(srv, r["gen"], prompt, r["reasoning"], r["seed"])
    meta.update(m)
    meta["finished"] = time.time()
    write(os.path.join(d, "answer.md"), answer)
    write(os.path.join(d, "thinking.md"), thinking)
    jdump(lps, os.path.join(d, "logprobs.json"))
    jdump(meta, os.path.join(d, "meta.json"))
    db.set_status(con, rid, "gen", "done")
    db.event(con, WHO, "gen done %s (%.0fs, %s ans tok, %.1f tok/s)" % (rid, meta.get("wall_s", 0),
                                                                       meta.get("answer_tokens"), meta.get("gen_tok_per_s") or 0))
    return answer


def orig_prompt(o):
    p = passages()[o["passage"]]
    pr = generation_prompt(p, corpus(), "G3")
    if o["web"]:
        nb = web.notes_block(o["run_id"])
        if nb:
            pr = pr + "\n\n" + nb + WEB_NOTE
    return pr


def wait_extracted(con, rids, what):
    while True:
        left = [r for r in rids if status(con, r, "v1_status") != "done"]
        failed = [r for r in left if status(con, r, "extract_status") == "failed"]
        if not left or len(failed) == len(left):
            return failed
        db.event(con, WHO, "waiting for extraction (%s): %d left" % (what, len(left))) if int(time.time()) % 1800 < 60 else None
        time.sleep(60)


# ------------------------------------------------------------------ cut
def _sentences(text):
    return [m for m in re.finditer(r"[^\n.!?]*(?:[.!?]+|\n|$)", text) if m.group(0).strip()]


def cut(answer, claims, flagged_ids):
    """Remove each sentence that holds a flagged claim's span; returns (new_answer, cut_ids)."""
    from .detectors import locate
    kill, cut_ids = [], []
    for c in claims:
        if c["id"] not in flagged_ids:
            continue
        loc = locate(answer, c.get("span"), c.get("orig"))
        if not loc:
            continue
        a, b = loc
        for m in _sentences(answer):
            if m.end() > a and m.start() < b:
                kill.append((m.start(), m.end()))
        cut_ids.append(c["id"])
    if not kill:
        return answer, cut_ids
    kill.sort()
    out, pos = [], 0
    for s, e in kill:
        if s >= pos:
            out.append(answer[pos:s])
            pos = e
        else:
            pos = max(pos, e)
    out.append(answer[pos:])
    return re.sub(r"[ \t]{2,}", " ", "".join(out)), cut_ids


# ------------------------------------------------------------------ phases
def run_gen(gen, con, pool):
    os_ = originals(gen)
    ps = passages()
    for o in os_:
        register(con, o)
        for s in resamples(o):
            register(con, s, extract="na", judge="na")
    # 1) web research for web-on originals
    for o in [o for o in os_ if o["web"]]:
        if os.path.exists(os.path.join(RUNS, o["run_id"], "web.json")):
            continue
        wait_for_thor(con, "web plan " + o["run_id"])
        os.makedirs(os.path.join(RUNS, o["run_id"]), exist_ok=True)
        q = web.plan_queries(pool.get(gen), base_prompt(ps[o["passage"]]), seed=o["seed"] % 100000)
        rec = web.research(o["run_id"], q)
        db.event(con, WHO, "web %s: %d queries, %d pages (%s)" % (o["run_id"], len(q), len(rec["pages"]),
                                                                   ", ".join(sorted({p["domain"] for p in rec["pages"]}))[:200]))
    # 2) originals, 3) V2 resamples
    for o in os_:
        jobs = [o] + resamples(o)
        for j in jobs:
            if status(con, j["run_id"], "gen_status") == "done":
                continue
            wait_for_thor(con, "gen " + j["run_id"])
            try:
                gen_files(con, pool.get(gen), j, orig_prompt(o),
                          extra_meta={"web": o["web"], "web_run": o["run_id"] if o["web"] else None,
                                      "parent": j.get("parent")})
            except Exception:
                db.event(con, WHO, "gen FAILED %s: %s" % (j["run_id"], traceback.format_exc()[-600:]))
                db.set_status(con, j["run_id"], "gen", "failed")
    pool.stop()
    # 4) extraction + V1 of originals runs in the labelcli worker; wait for it
    failed = wait_extracted(con, [o["run_id"] for o in os_], gen)
    if failed:
        db.event(con, WHO, "extraction failed for %s" % failed)
    # 5) V5 (gemma cross-family) and V2 scoring on originals
    from .detectors import run_v5_batch, run_v2_score
    rows = [dict(con.execute("SELECT * FROM runs WHERE run_id=?", (o["run_id"],)).fetchone()) for o in os_
            if not os.path.exists(os.path.join(RUNS, o["run_id"], "v5.json")) and o["run_id"] not in failed]
    run_v5_batch(con, rows, ps, corpus())
    for o in (os_ if CFG["v2_n"] else []):
        if o["run_id"] in failed or os.path.exists(os.path.join(RUNS, o["run_id"], "v2.json")):
            continue
        run_v2_score(con, dict(con.execute("SELECT * FROM runs WHERE run_id=?", (o["run_id"],)).fetchone()), ps, corpus())
    # 6) flags -> repair (own packet) -> delivered, for each V2 x rule cell
    th = jload(CFG["th_path"] or fusion.TH_V0)
    for o in os_:
        if o["run_id"] in failed:
            continue
        p = ps[o["passage"]]
        ocl = jload(os.path.join(RUNS, o["run_id"], "claims.json"), {}).get("claims", [])
        ans = open(os.path.join(RUNS, o["run_id"], "answer.md"), encoding="utf-8").read()
        for c in cells(o):
            did = c["run_id"] + "__delivered"
            if status(con, did, "gen_status") == "done":
                continue
            dets = ["V3", "V5"] + (["V2"] if c["v2"] else [])
            fl = fusion.flags(o["run_id"], gen, th, c["rule"], dets)
            byid = {x["id"]: x for x in ocl}
            dr = dict(c, run_id=did, block=CFG["dblock"], parent=o["run_id"], seed=(o["seed"] or 1) + 7)
            register(con, dr, judge="na")
            dd = os.path.join(RUNS, did)
            os.makedirs(dd, exist_ok=True)
            jdump({"rule": c["rule"], "dets": dets, "thresholds": CFG["th_label"], "flagged": [{"id": i, "why": w} for i, w in fl],
                   "n_claims": len(ocl)}, os.path.join(dd, "repair_flags.json"))
            if not fl:
                for f in ("answer.md", "thinking.md", "logprobs.json", "prompt.txt"):
                    shutil.copy2(os.path.join(RUNS, o["run_id"], f), os.path.join(dd, f))
                m = jload(os.path.join(RUNS, o["run_id"], "meta.json"), {})
                m.update(run_id=did, repair="no flags; delivered unchanged", parent=o["run_id"])
                jdump(m, os.path.join(dd, "meta.json"))
                db.set_status(con, did, "gen", "done")
                continue
            lines = ["%d. %s  [%s]" % (i + 1, byid[cid].get("claim"), why) for i, (cid, why) in enumerate(fl) if cid in byid]
            prompt = REPAIR_TMPL.format(task=base_prompt(p), answer=ans, flags="\n".join(lines),
                                        packet=own_packet(dict(o, web_run=o["run_id"] if o["web"] else None), p, corpus()))
            wait_for_thor(con, "repair " + did)
            try:
                gen_files(con, pool.get(gen), dr, prompt, out_id=did,
                          extra_meta={"parent": o["run_id"], "v2": c["v2"], "rule": c["rule"], "n_flagged": len(fl),
                                      "web": o["web"], "web_run": o["run_id"] if o["web"] else None})
            except Exception:
                db.event(con, WHO, "repair FAILED %s: %s" % (did, traceback.format_exc()[-600:]))
                db.set_status(con, did, "gen", "failed")
    pool.stop()
    # 7) extraction + V1 of delivered (labelcli worker), then V5 on delivered
    dids = [c["run_id"] + "__delivered" for o in os_ if o["run_id"] not in failed for c in cells(o)]
    dfailed = wait_extracted(con, dids, gen + " delivered")
    rows = [dict(con.execute("SELECT * FROM runs WHERE run_id=?", (d,)).fetchone()) for d in dids
            if d not in dfailed and not os.path.exists(os.path.join(RUNS, d, "v5.json"))]
    run_v5_batch(con, rows, ps, corpus())
    # 8) re-check with the cell's rule (V2 not available on delivered answers), cut, register finals for judging
    for o in os_:
        if o["run_id"] in failed:
            continue
        for c in cells(o):
            did = c["run_id"] + "__delivered"
            fid = c["run_id"] + "__final"
            if did in dfailed or status(con, fid, "gen_status") == "done":
                continue
            dcl = jload(os.path.join(RUNS, did, "claims.json"), {}).get("claims", [])
            fl = fusion.flags(did, gen, th, c["rule"], ["V3", "V5"])
            dans = open(os.path.join(RUNS, did, "answer.md"), encoding="utf-8").read()
            fans, cut_ids = cut(dans, dcl, {i for i, _ in fl})
            fd = os.path.join(RUNS, fid)
            os.makedirs(fd, exist_ok=True)
            write(os.path.join(fd, "answer.md"), fans)
            keep = [x for x in dcl if x["id"] not in set(cut_ids)]
            cj = jload(os.path.join(RUNS, did, "claims.json"))
            cj.update(run_id=fid, claims=keep, cut_from=did, n_cut=len(cut_ids))
            jdump(cj, os.path.join(fd, "claims.json"))
            v1 = [x for x in (jload(os.path.join(RUNS, did, "v1.json"), []) or []) if x.get("id") not in set(cut_ids)]
            jdump(v1, os.path.join(fd, "v1.json"))
            m = jload(os.path.join(RUNS, did, "meta.json"), {})
            m.update(run_id=fid, final_of=did, recheck_flagged=len(fl), cut_claims=len(cut_ids),
                     cut_chars=len(dans) - len(fans))
            jdump(m, os.path.join(fd, "meta.json"))
            register(con, dict(c, run_id=fid, block=CFG["fblock"], parent=did), gen_status="done",
                     extract="done", judge="pending")
            db.set_status(con, fid, "v1", "done")
            db.event(con, WHO, "final %s: %d re-flagged, %d cut" % (fid, len(fl), len(cut_ids)))
    db.event(con, WHO, "generator %s done" % gen)


def main():
    gens = sys.argv[sys.argv.index("--gens") + 1].split(",") if "--gens" in sys.argv else GENS
    if "--dry" in sys.argv:
        for g in gens:
            for o in originals(g):
                print(o["ord"], o["run_id"], o["seed"])
        return
    if not os.path.exists(CFG["th_path"] or fusion.TH_V0):
        raise SystemExit("thresholds file missing")
    con = db.connect()
    pool = Server()
    try:
        for g in gens:
            run_gen(g, con, pool)
    finally:
        pool.stop()
        harness.kill_llama_servers()
    db.event(con, WHO, "Stage C finished")
    open(os.path.join(STATE, CFG["done"]), "w").write(time.ctime())


if __name__ == "__main__":
    main()
