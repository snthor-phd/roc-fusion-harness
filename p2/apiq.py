"""API daemon: claim extraction (Opus 4.8) -> V1 -> judging (Opus 5.5), Batch API, hard $50 cap,
labeling in D22 priority order, V1 known-answer gate and the one-time rubric gate (plan 12.1)."""
import json
import os
import subprocess
import sys
import time
import traceback

from . import db, api, extract, judge
from .common import RUNS, STATE, REPO, jdump, jload
from .design import LABEL_PRI
from .v1 import run_v1, V1_VERSION

WHO = "api"
PASSAGES = os.path.join(STATE, "passages.json")
V1_FLAG = os.path.join(STATE, "V1_TESTS_PASSED")
HOLD = os.path.join(STATE, "JUDGE_HOLD")
GATE = os.path.join(STATE, "rubric_gate.json")
DEFAULT_COST_PER_RUN = 0.15
BATCH_MAX = 40

_corpus = None


def corpus():
    global _corpus
    if _corpus is None:
        from .texts import load_corpus
        _corpus = load_corpus()
    return _corpus


def passages():
    return {p["id"]: p for p in jload(PASSAGES)}


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def label_priority(r):
    if r["repair"] == "on":
        return 0  # repair chain needs its claims early (V1/V5 flags)
    return LABEL_PRI.get(r["block"], 9)


def run_v1_tests(con):
    try:
        out = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_v1.py"], cwd=REPO,
                             capture_output=True, text=True, timeout=600)
        ok = out.returncode == 0
        tail = (out.stdout + out.stderr).strip().splitlines()[-1:]
    except Exception as e:
        ok, tail = False, [repr(e)]
    if ok:
        jdump({"ts": time.time(), "result": tail, "v1": V1_VERSION}, V1_FLAG)
    elif os.path.exists(V1_FLAG):
        os.remove(V1_FLAG)
    db.event(con, WHO, "V1 known-answer tests: %s %s" % ("PASS" if ok else "FAIL", tail))
    return ok


def in_flight(con, kind):
    ids = set()
    for b in con.execute("SELECT custom_ids FROM batches WHERE kind=? AND status IN ('submitted','in_progress')", (kind,)):
        ids.update(json.loads(b["custom_ids"]).values())
    return ids


def attempts(kind, run_id, bump=False):
    p = os.path.join(STATE, "api_attempts.json")
    a = jload(p, {})
    k = kind + ":" + run_id
    if bump:
        a[k] = a.get(k, 0) + 1
        jdump(a, p)
    return a.get(k, 0)


# ------------------------------------------------------------------ result handlers
def make_handler(con):
    ps = passages()

    def handler(kind, cid, result):
        run_id = cid.split("__", 1)[1]
        if kind == "extract":
            data = api.tool_input(result)
            if data and isinstance(data.get("claims"), str):  # model occasionally stringifies the array
                try:
                    data["claims"] = json.loads(data["claims"])
                except ValueError:
                    pass
            if data is None or not isinstance(data.get("claims"), list):
                db.event(con, WHO, "extract failed %s: %s" % (run_id, json.dumps(result.get("error") or result.get("type"))[:400]))
                n = attempts("extract", run_id, bump=True)
                db.set_status(con, run_id, "extract", "failed" if n >= 2 else "pending", "extract result: %s" % result.get("type"))
                return
            meta = {"usage": result["message"].get("usage"), "model": result["message"].get("model"),
                    "stop": result["message"].get("stop_reason")}
            extract.save(run_id, data, meta)
            db.set_status(con, run_id, "extract", "done")
        elif kind == "supp":
            data = api.tool_input(result)
            ctx = jload(os.path.join(RUNS, run_id, "supp_ctx.json"))
            if data is None or not ctx:
                db.event(con, WHO, "supplement failed %s" % run_id)
                return
            cl = {c["id"]: c for c in jload(os.path.join(RUNS, run_id, "claims.json"))["claims"]}
            judge.merge_supplement(run_id, data.get("labels"), ctx["key_map"], ctx["packet_text"], cl)
            ctx["done"] = time.time()
            jdump(ctx, os.path.join(RUNS, run_id, "supp_ctx.json"))
        elif kind == "judge":
            data = api.tool_input(result)
            ctx = jload(os.path.join(RUNS, run_id, "judge_ctx.json"))
            if data is None or not ctx:
                db.event(con, WHO, "judge failed %s: %s" % (run_id, json.dumps(result.get("error") or result.get("type"))[:400]))
                n = attempts("judge", run_id, bump=True)
                db.set_status(con, run_id, "judge", "failed" if n >= 2 else "pending", "judge result: %s" % result.get("type"))
                return
            cl = jload(os.path.join(RUNS, run_id, "claims.json"))["claims"]
            v1 = {x["id"]: x for x in jload(os.path.join(RUNS, run_id, "v1.json"))}
            res = judge.assemble(run_id, cl, v1, data.get("labels"), ctx["key_map"], ctx["seeds"], ctx["packet_text"])
            res_meta = {"usage": result["message"].get("usage"), "model": result["message"].get("model"),
                        "stop": result["message"].get("stop_reason")}
            jdump(res_meta, os.path.join(RUNS, run_id, "judge_meta.json"))
            db.set_status(con, run_id, "judge", "done")
            # claims V1 decided when the request was built but no longer decides (V1 upgraded meanwhile)
            need = [x["id"] for x in res["labels"] if x.get("source") == "missing" and not x.get("seeded")]
            if need:
                sub = [c for c in cl if c["id"] in set(need)]
                params, key_map, _, packet_text, _ = judge.build_request(
                    run_id, ps[jload(os.path.join(RUNS, run_id, "meta.json"))["passage"]], corpus(), sub, {}, supplement=True)
                jdump({"key_map": key_map, "packet_text": packet_text, "params": params, "n": len(sub)},
                      os.path.join(RUNS, run_id, "supp_ctx.json"))
    return handler


# ------------------------------------------------------------------ steps
def step_v1(con):
    ps = passages()
    rows = con.execute("SELECT * FROM runs WHERE extract_status='done' AND v1_status='pending'").fetchall()
    for r in rows:
        try:
            cl = jload(os.path.join(RUNS, r["run_id"], "claims.json"))["claims"]
            res = run_v1(corpus(), ps[r["passage"]], cl)
            jdump(res, os.path.join(RUNS, r["run_id"], "v1.json"))
            jdump({"version": V1_VERSION, "ts": time.time()}, os.path.join(RUNS, r["run_id"], "v1_meta.json"))
            db.set_status(con, r["run_id"], "v1", "done")
        except Exception as e:
            db.event(con, WHO, "v1 failed %s: %s" % (r["run_id"], traceback.format_exc()[-800:]))
            db.set_status(con, r["run_id"], "v1", "failed", repr(e)[:300])


def avg_cost_per_run(con):
    rows = con.execute("SELECT custom_id, cost FROM ledger").fetchall()
    per = {}
    for r in rows:
        rid = r["custom_id"].split("__", 1)[1]
        per[rid] = per.get(rid, 0) + r["cost"]
    done = [rid for (rid,) in con.execute("SELECT run_id FROM runs WHERE judge_status='done'").fetchall()]
    vals = [per[r] for r in done if r in per]
    return (sum(vals) / len(vals)) if len(vals) >= 5 else DEFAULT_COST_PER_RUN


def higher_priority_backlog(con, pri):
    """Number of runs in strictly higher-priority labeling blocks not yet labeled (excluding failures)."""
    rows = con.execute("""SELECT block, repair FROM runs WHERE judge_status IN ('pending') AND gen_status IN ('pending','running','done')
                          AND block IN ('core','bridge','repair','big')""").fetchall()
    return sum(1 for r in rows if label_priority(r) < pri and label_priority(r) > 0)


def step_extract(con):
    if in_flight(con, "extract"):
        return
    rows = con.execute("SELECT * FROM runs WHERE gen_status='done' AND extract_status='pending'").fetchall()
    rows = [r for r in rows if r["block"] != "v2gen"]
    if not rows:
        return
    rows.sort(key=lambda r: (label_priority(r), r["ord"]))
    ps = passages()
    items = []
    unit = avg_cost_per_run(con)
    room = api.headroom(con)
    for r in rows[:BATCH_MAX]:
        pri = label_priority(r)
        if pri > 1:
            # D22: keep enough budget for higher-priority runs still to come
            if room - higher_priority_backlog(con, pri) * unit < unit:
                continue
        ans = read(os.path.join(RUNS, r["run_id"], "answer.md"))
        if not ans.strip():
            extract.save(r["run_id"], {"claims": [], "sections_answered": []}, {"note": "empty answer"})
            db.set_status(con, r["run_id"], "extract", "done")
            continue
        items.append(("x__" + r["run_id"], extract.build_request(r, ps[r["passage"]], ans)))
    if items:
        api.submit(con, "extract", extract.EXTRACT_MODEL, items)


def rubric_gate(con):
    """Plan 12.1: rubric may be revised at most once, after the first 20 answers, only if citation-verification
    failures exceed 10%. Returns how many more judge requests may be sent now (None = unlimited)."""
    g = jload(GATE, {})
    if g.get("decision") in ("continue", "revised"):
        return None
    done = [r["run_id"] for r in con.execute("SELECT run_id FROM runs WHERE judge_status='done' ORDER BY updated").fetchall()]
    if len(done) < 20:
        return max(0, 20 - len(done) - len(in_flight(con, "judge")))
    first = done[:20]
    chk = fail = 0
    for rid in first:
        lab = jload(os.path.join(RUNS, rid, "labels.json"), {})
        chk += lab.get("citation_checked", 0)
        fail += lab.get("citation_failed", 0)
    rate = fail / chk if chk else 0.0
    if rate > 0.10:
        jdump({"evaluated": time.time(), "rate": rate, "checked": chk, "failed": fail, "decision": "hold-for-revision"}, GATE)
        if not os.path.exists(HOLD):
            open(HOLD, "w").write("citation failure rate %.3f > 0.10 after 20 answers" % rate)
            db.event(con, WHO, "RUBRIC GATE: citation failures %.1f%% > 10%% after 20 answers; judging held for one revision" % (100 * rate))
        return 0
    jdump({"evaluated": time.time(), "rate": rate, "checked": chk, "failed": fail, "decision": "continue"}, GATE)
    db.event(con, WHO, "rubric gate passed: citation failures %.1f%% over first 20 answers" % (100 * rate))
    return None


def step_judge(con):
    if not os.path.exists(V1_FLAG) or os.path.exists(HOLD):
        return
    if in_flight(con, "judge"):
        return
    allow = rubric_gate(con)
    if allow == 0:
        return
    rows = con.execute("SELECT * FROM runs WHERE v1_status='done' AND judge_status='pending'").fetchall()
    rows = [r for r in rows if r["block"] != "v2gen"]
    if not rows:
        return
    rows.sort(key=lambda r: (LABEL_PRI.get(r["block"], 9), r["ord"]))
    if allow is not None:
        rows = rows[:allow]
    ps = passages()
    items = []
    unit = avg_cost_per_run(con)
    room = api.headroom(con)
    for r in rows[:BATCH_MAX]:
        pri = LABEL_PRI.get(r["block"], 9)
        if pri > 1 and room - higher_priority_backlog(con, pri) * unit < unit:
            continue
        cl = jload(os.path.join(RUNS, r["run_id"], "claims.json"))["claims"]
        v1 = {x["id"]: x for x in jload(os.path.join(RUNS, r["run_id"], "v1.json"))}
        p = ps[r["passage"]]
        try:
            params, key_map, seeds, packet_text, meta = judge.build_request(r["run_id"], p, corpus(), cl, v1)
        except Exception:
            # one malformed run must not stall the queue (12-13 Oct: 9 h stall on an int span)
            db.event(con, WHO, "judge build failed %s: %s" % (r["run_id"], traceback.format_exc()[-600:]))
            db.set_status(con, r["run_id"], "judge", "failed", "build_request error")
            continue
        if meta["n_undecided"] + meta["n_audit"] + meta["n_seeded"] == 0:
            judge.assemble(r["run_id"], cl, v1, [], {}, [], "")
            db.set_status(con, r["run_id"], "judge", "done")
            continue
        jdump({"key_map": key_map, "seeds": seeds, "packet_text": packet_text, "meta": meta},
              os.path.join(RUNS, r["run_id"], "judge_ctx.json"))
        items.append(("j__" + r["run_id"], params))
    if items:
        api.submit(con, "judge", judge.JUDGE_MODEL, items)


def step_v1_upgrade(con):
    """After a V1 version change: recompute V1 for every extracted run; for judged runs re-finalize labels and queue a
    supplemental judge request for claims V1 no longer decides (DECISIONS J24)."""
    ps = passages()
    rows = con.execute("SELECT * FROM runs WHERE v1_status='done'").fetchall()
    for r in rows:
        d = os.path.join(RUNS, r["run_id"])
        meta_p = os.path.join(d, "v1_meta.json")
        if (jload(meta_p, {}) or {}).get("version") == V1_VERSION:
            continue
        cl = jload(os.path.join(d, "claims.json"))["claims"]
        old = os.path.join(d, "v1.json")
        if os.path.exists(old) and not os.path.exists(os.path.join(d, "v1_v1.0.json")):
            os.replace(old, os.path.join(d, "v1_v1.0.json"))
        res = run_v1(corpus(), ps[r["passage"]], cl)
        jdump(res, old)
        jdump({"version": V1_VERSION, "ts": time.time()}, meta_p)
        if r["judge_status"] == "done" and os.path.exists(os.path.join(d, "labels.json")):
            need = judge.refinalize(r["run_id"], {x["id"]: x for x in res})
            if need:
                sub = [c for c in cl if c["id"] in set(need)]
                params, key_map, _, packet_text, _ = judge.build_request(
                    r["run_id"], ps[r["passage"]], corpus(), sub, {}, supplement=True)
                jdump({"key_map": key_map, "packet_text": packet_text, "params": params, "n": len(sub)},
                      os.path.join(d, "supp_ctx.json"))


def step_supp(con):
    if in_flight(con, "supp"):
        return
    items = []
    for r in con.execute("SELECT run_id FROM runs WHERE judge_status='done'").fetchall():
        ctx = jload(os.path.join(RUNS, r["run_id"], "supp_ctx.json"))
        if ctx and not ctx.get("done") and ctx.get("tries", 0) < 2:
            ctx["tries"] = ctx.get("tries", 0) + 1
            jdump(ctx, os.path.join(RUNS, r["run_id"], "supp_ctx.json"))
            items.append(("s__" + r["run_id"], ctx["params"]))
    if items:
        api.submit(con, "supp", judge.JUDGE_MODEL, items[:BATCH_MAX])


def write_status(con):
    st = {"ts": time.time(), "spent": round(db.spent(con), 4), "reserved": round(db.reserved(con), 4),
          "cap": api.CAP_USD, "v1_tests": os.path.exists(V1_FLAG), "judge_hold": os.path.exists(HOLD),
          "gate": jload(GATE, {})}
    for s in ("gen", "extract", "v1", "judge", "v5", "repair", "v2"):
        st[s] = {r[0]: r[1] for r in con.execute("SELECT %s_status, COUNT(*) FROM runs WHERE run_id NOT LIKE '__flag%%' GROUP BY 1" % s)}
    jdump(st, os.path.join(STATE, "status_api.json"))


def main():
    con = db.connect()
    db.event(con, WHO, "api daemon start pid %d" % os.getpid())
    run_v1_tests(con)
    handler = make_handler(con)
    while True:
        jdump({"pid": os.getpid(), "ts": time.time()}, os.path.join(STATE, "heartbeat_api.json"))
        if os.path.exists(os.path.join(STATE, "RELOAD_API")):
            os.remove(os.path.join(STATE, "RELOAD_API"))
            db.event(con, WHO, "reload requested; exiting for launchd restart")
            sys.exit(0)
        if not os.path.exists(os.path.join(STATE, "PAUSE_API")):
            for step in (lambda: api.poll(con, handler), lambda: step_v1(con), lambda: step_v1_upgrade(con),
                         lambda: step_extract(con), lambda: step_judge(con), lambda: step_supp(con),
                         lambda: write_status(con)):
                try:
                    step()
                except Exception:
                    db.event(con, WHO, "step error: %s" % traceback.format_exc()[-1200:])
        time.sleep(120)


if __name__ == "__main__":
    main()
