"""Local-compute daemon: generation (cold protocol), V5 local judge, repair loop, V2 scoring.

Runs under launchd (KeepAlive). One task at a time. Between tasks it yields to Thor's workloads and honours
state/PAUSE_LOCAL and state/RELOAD_LOCAL (exit so launchd restarts with new code).
"""
import os
import sys
import time
import traceback

from . import db
from .common import RUNS, STATE, jdump, jload
from .design import PRI
from . import harness

PASSAGES = os.path.join(STATE, "passages.json")
WHO = "local"


def passages():
    return {p["id"]: p for p in jload(PASSAGES)}


_corpus = None


def corpus():
    global _corpus
    if _corpus is None:
        from .texts import load_corpus
        _corpus = load_corpus()
    return _corpus


def run_dir(run_id):
    d = os.path.join(RUNS, run_id)
    os.makedirs(d, exist_ok=True)
    return d


def write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text or "")


# ------------------------------------------------------------------ task selection
def next_gen(con):
    blocked = con.execute("SELECT 1 FROM runs WHERE run_id='__flag_122b_blocked'").fetchone()
    rows = con.execute("""SELECT * FROM runs WHERE gen_status='pending' AND attempts < 2
                          AND block IN ('core','bridge','repair','big','v2gen')""").fetchall()
    rows = [r for r in rows if not (blocked and r["block"] == "big")]
    # bridge cells need Ollama 0.40.2 model; repair cells and delivered answers are handled by the repair task
    rows = [r for r in rows if r["repair"] != "delivered"]
    if not rows:
        return None, None
    rows = sorted(rows, key=lambda r: (PRI[r["block"]], r["ord"]))
    r = rows[0]
    return PRI[r["block"]], r


def next_repair(con):
    r = con.execute("""SELECT * FROM runs WHERE repair='on' AND repair_status='pending' AND gen_status='done'
                       AND v1_status='done' AND v5_status='done' ORDER BY ord LIMIT 1""").fetchone()
    return (PRI["repair_chain"], r) if r else (None, None)


def v5_pending(con):
    rows = con.execute("""SELECT * FROM runs WHERE extract_status='done' AND v5_status='pending'""").fetchall()
    out = []
    for r in rows:
        if r["block"] == "v2gen":
            continue
        if r["repair"] == "on":
            pri = PRI["repair_chain"]
        elif r["block"] == "big" or r["repair"] == "delivered":
            pri = PRI["v5_late"]
        else:
            pri = PRI["v5"]
        out.append((pri, r))
    return sorted(out, key=lambda x: (x[0], x[1]["ord"]))


def next_v2score(con):
    rows = con.execute("""SELECT * FROM runs WHERE v2_status='pending' AND extract_status='done'""").fetchall()
    for r in sorted(rows, key=lambda r: r["ord"]):
        kids = con.execute("SELECT gen_status FROM runs WHERE parent=? AND block='v2gen'", (r["run_id"],)).fetchall()
        if kids and all(k["gen_status"] in ("done", "failed") for k in kids):
            return PRI["v2score"], r
    return None, None


# ------------------------------------------------------------------ generation
def do_gen(con, r, prompt_override=None, out_id=None):
    ps = passages()
    p = ps[r["passage"]]
    from .prompts import generation_prompt
    rid = out_id or r["run_id"]
    d = run_dir(rid)
    prompt = prompt_override or generation_prompt(p, corpus(), r["grounding"])
    write(os.path.join(d, "prompt.txt"), prompt)
    cfg = harness.GENERATORS[r["gen"]]
    meta = {"run_id": rid, "block": r["block"], "gen": r["gen"], "model_label": cfg["label"],
            "stack": cfg["stack"], "passage": p["id"], "lang": p["lang"], "grounding": r["grounding"],
            "reasoning": r["reasoning"], "repair": r["repair"], "rep": r["rep"], "started": time.time(),
            "host": os.uname().nodename}
    if not out_id:
        db.set_status(con, rid, "gen", "running")
        con.execute("UPDATE runs SET attempts=attempts+1 WHERE run_id=?", (rid,))
    if cfg["stack"] == "S2":
        meta["engine"] = "llama-server " + harness.llama_version()
        meta["weights"] = os.path.basename(cfg["gguf"])
        meta["weight_hash"] = harness.file_hash(cfg["gguf"])
        srv = harness.LlamaServer(r["gen"])
        try:
            meta["load_s"] = round(srv.start(), 1)
            answer, thinking, lps, m = harness.generate_llama(srv, r["gen"], prompt, r["reasoning"], r["seed"])
        finally:
            srv.stop()
    else:
        harness.start_ollama()
        answer, thinking, lps, m = harness.generate_ollama(r["gen"], prompt, r["reasoning"], r["seed"])
    meta.update(m)
    meta["finished"] = time.time()
    meta["answer_chars"] = len(answer or "")
    write(os.path.join(d, "answer.md"), answer)
    write(os.path.join(d, "thinking.md"), thinking)
    jdump(lps, os.path.join(d, "logprobs.json"))
    jdump(meta, os.path.join(d, "meta.json"))
    if not out_id:
        db.set_status(con, rid, "gen", "done")
        db.event(con, WHO, "gen done %s (%.0fs, %s ans tok, %s think tok, %.1f tok/s)" % (
            rid, meta.get("wall_s", 0), meta.get("answer_tokens"), meta.get("think_tokens"), meta.get("gen_tok_per_s") or 0))
    if r["gen"] == "qwen35_122b":
        tps = meta.get("gen_tok_per_s") or 0
        if tps and tps < 1.0:
            con.execute("INSERT OR IGNORE INTO runs(run_id, block, gen_status) VALUES ('__flag_122b_blocked','flag','na')")
            db.event(con, WHO, "122B below 1 tok/s (%.2f): remaining 122B runs blocked (plan 12.1 rule)" % tps)
    return answer, meta


# ------------------------------------------------------------------ main loop
def main():
    con = db.connect()
    db.event(con, WHO, "local daemon start pid %d" % os.getpid())
    con.execute("UPDATE runs SET gen_status='pending' WHERE gen_status='running'")
    con.execute("UPDATE runs SET v5_status='pending' WHERE v5_status='running'")
    con.execute("UPDATE runs SET repair_status='pending' WHERE repair_status='running'")
    con.execute("UPDATE runs SET v2_status='pending' WHERE v2_status='running'")
    harness.kill_llama_servers()
    idle_logged = 0
    while True:
        jdump({"pid": os.getpid(), "ts": time.time()}, os.path.join(STATE, "heartbeat_local.json"))
        if os.path.exists(os.path.join(STATE, "RELOAD_LOCAL")):
            os.remove(os.path.join(STATE, "RELOAD_LOCAL"))
            db.event(con, WHO, "reload requested; exiting for launchd restart")
            harness.stop_ollama()
            sys.exit(0)
        if os.path.exists(os.path.join(STATE, "PAUSE_LOCAL")):
            time.sleep(30)
            continue
        busy = harness.thor_busy()
        if busy:
            if time.time() - idle_logged > 1800:
                db.event(con, WHO, "yielding to Thor's jobs: " + "; ".join(busy))
                idle_logged = time.time()
            harness.stop_ollama()
            time.sleep(60)
            continue
        cands = []
        pg, rg = next_gen(con)
        if rg is not None:
            cands.append((pg, "gen", rg))
        pr, rr = next_repair(con)
        if rr is not None:
            cands.append((pr, "repair", rr))
        v5 = v5_pending(con)
        if v5:
            cands.append((v5[0][0], "v5", v5[0][1]))
        pv, rv = next_v2score(con)
        if rv is not None:
            cands.append((pv, "v2score", rv))
        if not cands:
            time.sleep(60)
            continue
        cands.sort(key=lambda x: x[0])
        pri, kind, r = cands[0]
        try:
            if kind == "gen":
                if r["gen"] != "qwen36_mlx":
                    harness.stop_ollama()
                do_gen(con, r)
            elif kind == "v5":
                from . import detectors
                harness.stop_ollama()
                same = [x[1] for x in v5 if x[0] == pri]
                detectors.run_v5_batch(con, same[:12], passages(), corpus())
            elif kind == "repair":
                from . import repair
                harness.stop_ollama()
                repair.do_repair(con, r, passages(), corpus())
            elif kind == "v2score":
                from . import detectors
                harness.stop_ollama()
                detectors.run_v2_score(con, r, passages(), corpus())
        except Exception as e:
            tb = traceback.format_exc()
            db.event(con, WHO, "%s failed for %s: %s" % (kind, r["run_id"], tb[-1500:]))
            if kind == "gen":
                att = con.execute("SELECT attempts FROM runs WHERE run_id=?", (r["run_id"],)).fetchone()[0]
                db.set_status(con, r["run_id"], "gen", "failed" if att >= 2 else "pending", repr(e)[:500])
            else:
                stage = {"v5": "v5", "repair": "repair", "v2score": "v2"}[kind]
                fails = jload(os.path.join(STATE, "stage_fails.json"), {})
                k = kind + ":" + r["run_id"]
                fails[k] = fails.get(k, 0) + 1
                jdump(fails, os.path.join(STATE, "stage_fails.json"))
                db.set_status(con, r["run_id"], stage, "failed" if fails[k] >= 2 else "pending", repr(e)[:500])
            harness.kill_llama_servers()
            time.sleep(30)


if __name__ == "__main__":
    main()
