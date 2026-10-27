"""WP8 (Phase 0): verify-and-repair loop for the 2x2 repair mini-screen (J10).

Flag rule (provisional, pre-calibration): V1 = Contradicted OR V5 p_false >= 50. Flagged claims are sent back to the
same generator with the source packet for targeted correction; the delivered answer becomes a new run
(<run_id>__delivered) that is re-extracted and re-labeled to measure delivered HR.
"""
import os
import shutil
import time

from . import db
from .common import RUNS, jload, jdump
from .judge import full_packet

V5_THRESHOLD = 50

REPAIR_TMPL = """{task}

YOUR PREVIOUS ANSWER
{answer}

VERIFICATION. An automatic check flagged the statements below as probably wrong. For each one: correct it if the source materials settle the point; otherwise delete it or say plainly that it is uncertain. Do not add new manuscript sigla, counts, quotations or citations that the sources do not support. Keep everything that was not flagged as it was.

FLAGGED STATEMENTS
{flags}

SOURCE MATERIALS (authoritative for text, apparatus and morphology)
{packet}

Now write the complete corrected answer, with all eight sections."""


def own_packet(r, p, corpus):
    """Stage C: the cell's own grounding packet plus its own web notes (Phase 1 decision 1)."""
    from .prompts import grounding_packet
    from .web import notes_block
    parts = [grounding_packet(p, corpus, r["grounding"]) or "(no source packet at this grounding level)"]
    nb = notes_block(r.get("web_run") or r["run_id"])
    if nb:
        parts.append(nb)
    return "\n\n".join(parts)


def flagged(run_id):
    d = os.path.join(RUNS, run_id)
    cl = jload(os.path.join(d, "claims.json"), {}).get("claims", [])
    v1 = {x["id"]: x for x in jload(os.path.join(d, "v1.json"), [])}
    v5 = (jload(os.path.join(d, "v5.json"), {}) or {}).get("scores", {})
    out = []
    for c in cl:
        why = None
        if (v1.get(c["id"]) or {}).get("v1") == "C":
            why = "source check: " + v1[c["id"]]["why"]
        elif v5.get(c["id"], 0) >= V5_THRESHOLD:
            why = "independent checker: probability false %d%%" % v5[c["id"]]
        if why:
            out.append((c, why))
    return out


def do_repair(con, r, passages, corpus):
    from .localq import do_gen
    db.set_status(con, r["run_id"], "repair", "running")
    d = os.path.join(RUNS, r["run_id"])
    p = passages[r["passage"]]
    fl = flagged(r["run_id"])
    new_id = r["run_id"] + "__delivered"
    nd = os.path.join(RUNS, new_id)
    os.makedirs(nd, exist_ok=True)
    with open(os.path.join(d, "answer.md"), encoding="utf-8") as f:
        answer = f.read()
    from .prompts import base_prompt
    task = base_prompt(p)  # the packet is appended once below (avoids duplicating it for G2 runs)
    if not fl:
        shutil.copy(os.path.join(d, "answer.md"), os.path.join(nd, "answer.md"))
        jdump({"run_id": new_id, "repair": "no flags; answer delivered unchanged", "parent": r["run_id"]},
              os.path.join(nd, "meta.json"))
    else:
        lines = ["%d. %s  [%s]" % (i + 1, c.get("claim"), why) for i, (c, why) in enumerate(fl)]
        prompt = REPAIR_TMPL.format(task=task, answer=answer, flags="\n".join(lines), packet=own_packet(dict(r), p, corpus))
        rr = dict(r)
        rr["seed"] = (r["seed"] or 1) + 7
        do_gen(con, rr, prompt_override=prompt, out_id=new_id)
    jdump({"flagged": [{"id": c["id"], "why": why} for c, why in fl], "n_claims": len(jload(os.path.join(d, "claims.json"), {}).get("claims", [])),
           "rule": "V1==C or V5>=%d" % V5_THRESHOLD, "ts": time.time()}, os.path.join(nd, "repair_flags.json"))
    con.execute("""INSERT OR IGNORE INTO runs(run_id, block, gen, passage, lang, grounding, reasoning, repair, rep, seed, ord,
                   parent, gen_status, extract_status, v1_status, judge_status, v5_status, repair_status, v2_status, updated)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'done','pending','pending','pending','pending','na','na',?)""",
                (new_id, "repair", r["gen"], r["passage"], r["lang"], r["grounding"], r["reasoning"], "delivered", 1,
                 r["seed"], r["ord"], r["run_id"], time.time()))
    db.set_status(con, r["run_id"], "repair", "done")
    db.event(con, "local", "repair done %s: %d flagged claims" % (r["run_id"], len(fl)))
