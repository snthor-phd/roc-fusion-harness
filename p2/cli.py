"""Command line: python -m p2.cli <command>"""
import json
import os
import sys
import time

from .common import STATE, REPO, jdump, jload


def cmd_passages():
    from .texts import load_corpus
    from .sampler import build_passages, SEED
    c = load_corpus(force=True)
    ps, units = build_passages(c)
    out = os.path.join(STATE, "passages.json")
    if os.path.exists(out):
        old = jload(out)
        if [p["id"] for p in old] != [p["id"] for p in ps]:
            raise SystemExit("passages.json exists and differs; refusing to overwrite a frozen draw")
    jdump(ps, out)
    jdump(ps, os.path.join(REPO, "prereg", "passages_all_splits.json"))
    frame_stats = {"seed": SEED, "frame_units": len(units), "by_lang_genre": {},
                   "familiarity_source": "OpenBible.info cross-reference votes (CC-BY), mean votes per verse"}
    for u in units:
        k = u["lang"] + "|" + u["genre"]
        frame_stats["by_lang_genre"][k] = frame_stats["by_lang_genre"].get(k, 0) + 1
    jdump(frame_stats, os.path.join(REPO, "prereg", "frame_stats.json"))
    for p in ps:
        if p["split"] == "design":
            print(p["lang"], p["id"], p["genre"], p["dens_stratum"], p["fam_tercile"], p["focus_ref"],
                  p["lexemes"], "HALF" if p["half"] else "")


def cmd_initdb():
    from . import db
    from .design import phase0_runs, v2_runs
    ps = jload(os.path.join(STATE, "passages.json"))
    runs = phase0_runs(ps)
    v2 = v2_runs(runs)
    con = db.connect()
    n = db.insert_runs(con, runs + v2)
    jdump(runs, os.path.join(REPO, "prereg", "phase0_run_order.json"))
    print("inserted", n, "runs;", len(runs), "primary,", len(v2), "V2 resamples")


def cmd_status():
    from . import db
    con = db.connect()
    rows = con.execute("SELECT block, gen_status, COUNT(*) n FROM runs GROUP BY block, gen_status").fetchall()
    for r in rows:
        print(dict(r))
    for st in ("extract", "v1", "judge", "v5", "repair", "v2"):
        rows = con.execute("SELECT %s_status s, COUNT(*) n FROM runs GROUP BY s" % st).fetchall()
        print(st, {r["s"]: r["n"] for r in rows})
    print("spent $%.2f reserved $%.2f" % (db.spent(con), db.reserved(con)))
    for e in con.execute("SELECT * FROM events ORDER BY id DESC LIMIT 15").fetchall():
        print(time.strftime("%m-%d %H:%M", time.localtime(e["ts"])), e["who"], e["msg"][:300])


def cmd_prompt(arg):
    from .texts import load_corpus
    from .prompts import generation_prompt
    ps = {p["id"]: p for p in jload(os.path.join(STATE, "passages.json"))}
    pid, g = arg.split(",")
    print(generation_prompt(ps[pid], load_corpus(), g))


def cmd_smoke(gen, reasoning):
    """Tiny generation to verify template, thinking tags, cap logic and logprobs (not a design run)."""
    from . import harness
    prompt = "In one sentence, what does the Greek word λόγος mean? Then parse the form ἐγένετο."
    cfg = harness.GENERATORS[gen]
    old = (harness.THINK_CAP, harness.ANSWER_MAX)
    harness.THINK_CAP, harness.ANSWER_MAX = 300, 200
    try:
        if cfg["stack"] == "S2":
            s = harness.LlamaServer(gen)
            t = s.start()
            props = s.props()
            tpl = props.get("chat_template") or ""
            print("load_s", round(t, 1), "| template mentions enable_thinking:", "enable_thinking" in tpl)
            print("template tail (thinking on): ", repr(s.apply_template("hi", True)[-160:]))
            print("template tail (thinking off):", repr(s.apply_template("hi", False)[-160:]))
            if os.environ.get("SHOW_TEMPLATE"):
                print(tpl[-3000:])
            try:
                a, th, lps, m = harness.generate_llama(s, gen, prompt, reasoning, 1234)
            finally:
                s.stop()
        else:
            harness.start_ollama()
            a, th, lps, m = harness.generate_ollama(gen, prompt, reasoning, 1234)
        print("META", json.dumps(m, ensure_ascii=False)[:900])
        print("THINK[:300]", repr(th[:300]))
        print("ANSWER[:500]", repr(a[:500]))
        print("logprobs n", len(lps), lps[:4])
    finally:
        harness.THINK_CAP, harness.ANSWER_MAX = old


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "passages":
        cmd_passages()
    elif c == "initdb":
        cmd_initdb()
    elif c == "status":
        cmd_status()
    elif c == "prompt":
        cmd_prompt(sys.argv[2])
    elif c == "smoke":
        cmd_smoke(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit("unknown command")
