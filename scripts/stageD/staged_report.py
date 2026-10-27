"""Stage D analysis: the frozen harness (v1.0.0) on the 24 evaluation passages.

Pass rule (plan v2): one-sided 95% upper bound (passage-cluster bootstrap) on delivered HR < 30% in Greek AND Hebrew,
and judge-matched retention >= 0.70 in each language. Also reports draft HR, the paired drop, flags/cuts, judge checks.
Run p2.retention on all 24 (draft, final) pairs first.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from p2 import db  # noqa: E402
from p2.common import RUNS, STATE, jload, jdump  # noqa: E402
from p2.localq import passages  # noqa: E402
from p2.report import judge_quality  # noqa: E402

RNG = np.random.default_rng(20261022)
PS = passages()
EV = sorted(p for p, v in PS.items() if v["split"] == "evaluation")
G = "qwen35_122b"


def d_id(p):
    return "stageD__%s__%s__G3__web0" % (G, p)


def f_id(p):
    return d_id(p) + "__v20__2of3__final"


def counts(rid):
    h = n = disc = 0
    for l in jload(os.path.join(RUNS, rid, "labels.json"), {}).get("labels", []):
        if l.get("seeded") or l.get("type") == "attribution":
            continue
        if l.get("checkable"):
            n += 1
            h += bool(l.get("hallucinated"))
        else:
            disc += 1
    return h, n, disc


def boot(per):
    h = np.array([x[0] for x in per], float)
    n = np.array([x[1] for x in per], float)
    k = len(h)
    bs = [h[i].sum() / n[i].sum() for i in (RNG.integers(0, k, k) for _ in range(10000)) if n[i].sum()]
    return {"hr": round(h.sum() / n.sum(), 4), "lo95": round(float(np.percentile(bs, 2.5)), 4),
            "hi95": round(float(np.percentile(bs, 97.5)), 4), "ub95": round(float(np.percentile(bs, 95)), 4),
            "claims": int(n.sum()), "passages": k}


res = {"passages": len(EV), "missing": [], "draft": {}, "final": {}, "retention": {}, "paired_drop": {}}
done = [p for p in EV if os.path.exists(os.path.join(RUNS, f_id(p), "labels.json"))
        and os.path.exists(os.path.join(RUNS, d_id(p), "labels.json"))]
res["missing"] = [p for p in EV if p not in done]
for lang in ("all", "greek", "hebrew"):
    ps = [p for p in done if lang == "all" or PS[p]["lang"] == lang]
    res["draft"][lang] = boot([counts(d_id(p))[:2] for p in ps])
    res["final"][lang] = boot([counts(f_id(p))[:2] for p in ps])
    rets = [jload(os.path.join(RUNS, f_id(p), "retention.json")) for p in ps]
    rets = [r for r in rets if r]
    kept = sum(r["counts"]["kept"] + r["counts"]["reworded"] for r in rets)
    tot = sum(r["n"] for r in rets)
    res["retention"][lang] = {"retention": round(kept / tot, 4) if tot else None, "pairs": len(rets),
                              "contradicted": sum(r["counts"]["contradicted"] for r in rets)}
    diff = np.array([counts(f_id(p))[0] / max(1, counts(f_id(p))[1]) - counts(d_id(p))[0] / max(1, counts(d_id(p))[1]) for p in ps])
    bs = [diff[RNG.integers(0, len(diff), len(diff))].mean() for _ in range(10000)]
    res["paired_drop"][lang] = {"mean": round(float(diff.mean()), 4), "lo95": round(float(np.percentile(bs, 2.5)), 4),
                                "hi95": round(float(np.percentile(bs, 97.5)), 4)}
flags = [len(jload(os.path.join(RUNS, d_id(p) + "__v20__2of3__delivered", "repair_flags.json"), {}).get("flagged", [])) for p in done]
cuts = [jload(os.path.join(RUNS, f_id(p), "meta.json"), {}).get("cut_claims", 0) for p in done]
trunc = [p for p in done if jload(os.path.join(RUNS, d_id(p), "meta.json"), {}).get("answer_stop") == "limit"
         or jload(os.path.join(RUNS, d_id(p) + "__v20__2of3__delivered", "meta.json"), {}).get("answer_stop") == "limit"]
res["flags_per_answer"] = round(float(np.mean(flags)), 2) if flags else None
res["no_flag_answers"] = sum(1 for f in flags if f == 0)
res["cuts_per_answer"] = round(float(np.mean(cuts)), 2) if cuts else None
res["truncated"] = trunc
res["gate"] = {
    "greek_ub_lt_30": res["final"]["greek"]["ub95"] < 0.30,
    "hebrew_ub_lt_30": res["final"]["hebrew"]["ub95"] < 0.30,
    "greek_retention_ge_70": (res["retention"]["greek"]["retention"] or 0) >= 0.70,
    "hebrew_retention_ge_70": (res["retention"]["hebrew"]["retention"] or 0) >= 0.70,
}
res["gate"]["PASS"] = all(res["gate"].values()) and not res["missing"]
con = db.connect()
runs = [dict(r) for r in con.execute("SELECT * FROM runs WHERE block IN ('stageD','stageD_final')").fetchall()]
res["judge"] = judge_quality(runs)
jdump(res, os.path.join(STATE, "stageD_results.json"))
print(json.dumps(res, indent=1))
