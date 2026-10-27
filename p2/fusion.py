"""Live ROC fusion for the harness (Stage C).

Each detector is thresholded where its false-positive rate (FPR) on labeled claims is about 0.20 (the Phase 0
convention); the fused flag is OR ("any") or k-of-n ("2of3"). V1 contradictions are always flagged.
Scores are oriented so higher = more likely false: V3 = -mean log-prob, V5 = p_false (0-100), V2 = resample score.

thresholds_v0.json: calibrated before Stage C on Phase 0 + Stage B labels (V3 per generator, V5/V2 pooled).
Stage C's own labels re-tune the thresholds from its ROC curves (the Stage C gate), giving thresholds_v1.json.
"""
import math
import os

import numpy as np

from .common import RUNS, STATE, jload, jdump

FPR_TARGET = 0.20
TH_V0 = os.path.join(STATE, "thresholds_v0.json")


def threshold_at_fpr(scores, y, fpr=FPR_TARGET):
    s = np.asarray(scores, float)
    neg = np.sort(s[np.asarray(y) == 0])
    if len(neg) == 0:
        return None
    return float(neg[int(math.floor((1 - fpr) * (len(neg) - 1)))])


def claim_scores(run_id):
    """{claim_id: {"V1": 'S'|'C'|None, "V3": float|None, "V5": float|None, "V2": float|None}}"""
    from .detectors import v3_scores
    d = os.path.join(RUNS, run_id)
    cl = jload(os.path.join(d, "claims.json"), {}).get("claims", [])
    v1 = {x["id"]: x for x in jload(os.path.join(d, "v1.json"), []) or []}
    v5 = (jload(os.path.join(d, "v5.json"), {}) or {}).get("scores", {})
    v2 = (jload(os.path.join(d, "v2.json"), {}) or {}).get("scores", {})
    try:
        v3 = v3_scores(run_id)
    except Exception:
        v3 = {}
    out = {}
    for c in cl:
        i = c["id"]
        out[i] = {"V1": (v1.get(i) or {}).get("v1"),
                  "V3": (-v3[i]["mean_lp"]) if i in v3 else None,
                  "V5": float(v5[i]) if i in v5 else None,
                  "V2": (v2.get(i) or {}).get("score")}
    return out


def calibrate(runs_by_gen):
    """runs_by_gen: {gen: [run_id,...]} of judged runs. Returns thresholds dict."""
    rows = []
    for gen, rids in runs_by_gen.items():
        for rid in rids:
            labs = {l["id"]: l for l in jload(os.path.join(RUNS, rid, "labels.json"), {}).get("labels", [])}
            sc = claim_scores(rid)
            for cid, s in sc.items():
                l = labs.get(cid)
                if not l or l.get("seeded") or not l.get("checkable"):
                    continue
                rows.append(dict(s, gen=gen, y=int(l["hallucinated"])))
    th = {"fpr_target": FPR_TARGET, "V3": {}, "n": {}}
    for gen in runs_by_gen:
        sub = [r for r in rows if r["gen"] == gen and r["V3"] is not None]
        th["V3"][gen] = threshold_at_fpr([r["V3"] for r in sub], [r["y"] for r in sub])
        th["n"]["V3_" + gen] = len(sub)
    for det in ("V5", "V2"):
        sub = [r for r in rows if r[det] is not None]
        th[det] = threshold_at_fpr([r[det] for r in sub], [r["y"] for r in sub])
        th["n"][det] = len(sub)
    return th


def flags(run_id, gen, th, rule, dets):
    """Return [(claim_id, reason)] for claims to repair/cut. rule: 'any' or '2of3' (k = 2; with two detectors, both)."""
    sc = claim_scores(run_id)
    out = []
    for cid, s in sc.items():
        if s["V1"] == "C":
            out.append((cid, "source check contradicts"))
            continue
        if s["V1"] == "S":
            continue  # the source check confirms it
        hits = []
        for d in dets:
            t = th[d].get(gen) if isinstance(th.get(d), dict) else th.get(d)
            v = s.get(d)
            if t is not None and v is not None and v > t:
                hits.append(d)
        avail = [d for d in dets if s.get(d) is not None]
        need = 1 if rule == "any" else min(2, max(1, len(avail)))
        if len(hits) >= need:
            out.append((cid, "detectors " + "+".join(hits)))
    return out
