"""WP6 (Phase 0 subset): V3 token confidence, V5 local cross-family judge, V2 resampling consistency."""
import json
import math
import os
import re
import time

from . import db, harness
from .common import RUNS, jload, jdump
from .judge import full_packet

V5_JUDGE = {"qwen": "gemma4_26b", "gemma": "qwen36_35b"}  # cross-family: Gemma judges Qwen, Qwen judges Gemma
V2_SCORER = "gemma4_26b"


def family(gen):
    return harness.GENERATORS[gen]["family"]


# ------------------------------------------------------------------ V3
def _norm_ws(s):
    return re.sub(r"\s+", " ", s or "").strip()


def locate(answer, span, orig=None):
    """Return (start, end) char offsets of span in answer, or None."""
    if span:
        i = answer.find(span)
        if i >= 0:
            return i, i + len(span)
        # whitespace-insensitive search
        flat, idx = [], []
        for j, ch in enumerate(answer):
            if ch.isspace():
                if flat and flat[-1] == " ":
                    continue
                flat.append(" ")
            else:
                flat.append(ch)
            idx.append(j)
        f = "".join(flat)
        s = _norm_ws(span)
        k = f.find(s)
        if k >= 0 and k + len(s) - 1 < len(idx):
            return idx[k], idx[k + len(s) - 1] + 1
        # prefix match (extractor may truncate)
        if len(s) > 40:
            k = f.find(s[:40])
            if k >= 0 and k + 39 < len(idx):
                return idx[k], idx[min(len(idx) - 1, k + len(s) - 1)] + 1
    if orig:
        i = answer.find(orig)
        if i >= 0:
            return i, i + len(orig)
    return None


def v3_scores(run_id):
    """Per-claim mean and min token log-probability over the claim span (gray-box V3)."""
    d = os.path.join(RUNS, run_id)
    lps = jload(os.path.join(d, "logprobs.json"), [])
    cl = jload(os.path.join(d, "claims.json"), {}).get("claims", [])
    if not lps or not cl:
        return {}
    with open(os.path.join(d, "answer.md"), encoding="utf-8") as f:
        answer = f.read()
    offs, pos = [], 0
    text = "".join(t for t, _ in lps)
    # token stream may differ from saved answer by leading whitespace; align by offset of first 200 chars
    shift = answer.find(text[:200].strip()[:50]) - text.find(text[:200].strip()[:50]) if text.strip() else 0
    for t, lp in lps:
        offs.append((pos + shift, pos + shift + len(t), lp))
        pos += len(t)
    out = {}
    for c in cl:
        loc = locate(answer, c.get("span"), c.get("orig"))
        if not loc:
            continue
        a, b = loc
        vals = [lp for s, e, lp in offs if e > a and s < b and lp is not None and not (isinstance(lp, float) and math.isnan(lp))]
        if vals:
            out[c["id"]] = {"mean_lp": sum(vals) / len(vals), "min_lp": min(vals), "n_tok": len(vals)}
    return out


# ------------------------------------------------------------------ V5
V5_SCHEMA = {"type": "object", "properties": {"scores": {"type": "array", "items": {
    "type": "object", "properties": {"id": {"type": "string"}, "p_false": {"type": "integer", "minimum": 0, "maximum": 100}},
    "required": ["id", "p_false"]}}}, "required": ["scores"]}

V5_PROMPT = """You check claims from an AI-written Bible exegesis against source data. For each claim give p_false: your probability (0-100) that the claim is false or fabricated. Use the SOURCE PACKET where it applies (it is authoritative for the text, apparatus and morphology); otherwise use your own knowledge. Interpretive opinions get p_false 0-20 unless clearly wrong.

SOURCE PACKET
{packet}

CLAIMS
{claims}

Return JSON: {{"scores": [{{"id": "c001", "p_false": 10}}, ...]}} with one entry for every claim id."""


def _claim_lines(cl):
    out = []
    for c in cl:
        bits = [c["id"], c.get("type", "")]
        for f in ("ref", "orig", "parse"):
            if c.get(f):
                bits.append("%s=%s" % (f, c[f]))
        if c.get("witnesses"):
            bits.append("witnesses=" + ",".join(c["witnesses"]))
        bits.append(c.get("claim", ""))
        out.append(" | ".join(bits))
    return "\n".join(out)


def _json_from(text):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else None


def run_v5_batch(con, rows, passages, corpus):
    if not rows:
        return
    judge_gen = V5_JUDGE[family(rows[0]["gen"])]
    rows = [r for r in rows if V5_JUDGE[family(r["gen"])] == judge_gen]
    srv = harness.LlamaServer(judge_gen)
    srv.start()
    try:
        for r in rows:
            db.set_status(con, r["run_id"], "v5", "running")
            d = os.path.join(RUNS, r["run_id"])
            cl = jload(os.path.join(d, "claims.json"), {}).get("claims", [])
            if not cl:
                jdump({"judge": judge_gen, "scores": {}}, os.path.join(d, "v5.json"))
                db.set_status(con, r["run_id"], "v5", "done")
                continue
            p = passages[r["passage"]]
            packet = full_packet(p, corpus)
            scores = {}
            t0 = time.time()
            for i in range(0, len(cl), 40):  # chunks keep the output short and the grammar fast
                chunk = cl[i:i + 40]
                prompt = V5_PROMPT.format(packet=packet, claims=_claim_lines(chunk))
                tp = srv.apply_template(prompt, thinking=False)
                body = {"prompt": tp, "n_predict": 3000, "temperature": 0.0, "seed": 1, "cache_prompt": True,
                        "json_schema": V5_SCHEMA}
                res = harness.http_json(srv.base + "/completion", body, timeout=3600)
                try:
                    data = _json_from(res.get("content", "")) or {}
                except Exception:
                    data = {}
                for s in data.get("scores", []):
                    if s.get("id") in {c["id"] for c in chunk}:
                        scores[s["id"]] = max(0, min(100, int(s.get("p_false", 50))))
            jdump({"judge": judge_gen, "weights": harness.GENERATORS[judge_gen]["gguf"].split("/")[-1],
                   "scores": scores, "n_claims": len(cl), "wall_s": round(time.time() - t0, 1)}, os.path.join(d, "v5.json"))
            db.set_status(con, r["run_id"], "v5", "done")
            db.event(con, "local", "v5 done %s by %s (%d/%d scored)" % (r["run_id"], judge_gen, len(scores), len(cl)))
    finally:
        srv.stop()


# ------------------------------------------------------------------ V2
V2_SCHEMA = {"type": "object", "properties": {"verdicts": {"type": "array", "items": {
    "type": "object", "properties": {"id": {"type": "string"}, "v": {"type": "string", "enum": ["supports", "contradicts", "absent"]}},
    "required": ["id", "v"]}}}, "required": ["verdicts"]}

V2_PROMPT = """Below is ANSWER B, another answer to the same exegesis question, and a list of CLAIMS taken from a different answer (answer A). For each claim decide whether answer B supports it (states the same fact), contradicts it (states something incompatible), or is silent (absent). Judge only by what answer B says.

ANSWER B
{answer}

CLAIMS
{claims}

Return JSON: {{"verdicts": [{{"id": "c001", "v": "supports"}}, ...]}} with one entry for every claim id."""


def run_v2_score(con, r, passages, corpus):
    db.set_status(con, r["run_id"], "v2", "running")
    d = os.path.join(RUNS, r["run_id"])
    cl = jload(os.path.join(d, "claims.json"), {}).get("claims", [])
    kids = con.execute("SELECT run_id FROM runs WHERE parent=? AND block='v2gen' AND gen_status='done'", (r["run_id"],)).fetchall()
    tally = {c["id"]: {"supports": 0, "contradicts": 0, "absent": 0} for c in cl}
    if cl and kids:
        srv = harness.LlamaServer(V2_SCORER)
        srv.start()
        try:
            for k in kids:
                with open(os.path.join(RUNS, k["run_id"], "answer.md"), encoding="utf-8") as f:
                    ans = f.read()
                for i in range(0, len(cl), 40):
                    chunk = cl[i:i + 40]
                    tp = srv.apply_template(V2_PROMPT.format(answer=ans, claims=_claim_lines(chunk)), thinking=False)
                    body = {"prompt": tp, "n_predict": 3000, "temperature": 0.0, "seed": 1, "cache_prompt": True,
                            "json_schema": V2_SCHEMA}
                    res = harness.http_json(srv.base + "/completion", body, timeout=3600)
                    try:
                        data = _json_from(res.get("content", "")) or {}
                    except Exception:
                        data = {}
                    for v in data.get("verdicts", []):
                        if v.get("id") in tally and v.get("v") in tally[v["id"]]:
                            tally[v["id"]][v["v"]] += 1
        finally:
            srv.stop()
    n = len(kids)
    out = {cid: dict(t, n=n, score=((t["contradicts"] + 0.5 * t["absent"]) / n) if n else None) for cid, t in tally.items()}
    jdump({"scorer": V2_SCORER, "n_resamples": n, "scores": out}, os.path.join(d, "v2.json"))
    db.set_status(con, r["run_id"], "v2", "done")
    db.event(con, "local", "v2 scored %s with %d resamples" % (r["run_id"], n))
