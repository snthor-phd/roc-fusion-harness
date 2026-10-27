"""WP9 (Phase 0): analysis and daily summary.

python -m p2.report [--final]   -> writes state/report_<date>.json and reports/phase0_<date>.md, prints the summary.
"""
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np

from . import db
from .common import RUNS, STATE, REPO, P2, jload, jdump
from .detectors import v3_scores

sys.path.insert(0, os.path.join(REPO, "vendor"))
try:
    import bool_roc_bounds as BRB
except Exception:  # pragma: no cover
    BRB = None

GEN_LABEL = {"qwen38_27b": "Qwen3.8-27B", "qwen36_35b": "Qwen3.6-35B (GGUF)", "gemma4_26b": "gemma4:26b",
             "qwen35_122b": "Qwen3.5-122B", "qwen36_mlx": "Qwen3.6-35B (MLX)"}
RNG = np.random.default_rng(20261017)


def load_runs(con):
    rows = [dict(r) for r in con.execute("SELECT * FROM runs WHERE run_id NOT LIKE '__flag%'").fetchall()]
    return rows


def labels_for(run_id):
    lab = jload(os.path.join(RUNS, run_id, "labels.json"))
    return lab["labels"] if lab else None


def cell_counts(run_ids):
    """Per passage: (hallucinated, checkable, discarded) summed over runs."""
    per = defaultdict(lambda: [0, 0, 0])
    for rid, pid in run_ids:
        labs = labels_for(rid)
        if labs is None:
            continue
        for l in labs:
            if l.get("seeded"):
                continue
            f = l.get("final")
            if f in ("S", "C", "U"):
                per[pid][1] += 1
                if f in ("C", "U"):
                    per[pid][0] += 1
            elif f == "N":
                per[pid][2] += 1
    return per


def hr_summary(per, boot=2000):
    if not per:
        return None
    h = np.array([v[0] for v in per.values()], float)
    n = np.array([v[1] for v in per.values()], float)
    d = np.array([v[2] for v in per.values()], float)
    if n.sum() == 0:
        return None
    hr = h.sum() / n.sum()
    k = len(h)
    bs = []
    for _ in range(boot):
        idx = RNG.integers(0, k, k)
        if n[idx].sum() > 0:
            bs.append(h[idx].sum() / n[idx].sum())
    lo, hi = (np.percentile(bs, [2.5, 97.5]) if bs else (float("nan"), float("nan")))
    ub95 = np.percentile(bs, 95) if bs else float("nan")
    worst = (h.sum() + d.sum()) / (n.sum() + d.sum()) if (n.sum() + d.sum()) else float("nan")
    best = h.sum() / (n.sum() + d.sum()) if (n.sum() + d.sum()) else float("nan")
    return {"hr": hr, "ci": [float(lo), float(hi)], "ub95_one_sided": float(ub95), "passages": k,
            "checkable": int(n.sum()), "hallucinated": int(h.sum()), "discarded": int(d.sum()),
            "discard_rate": float(d.sum() / (n.sum() + d.sum())) if (n.sum() + d.sum()) else None,
            "hr_all_discards_false": worst, "hr_all_discards_true": best}


def tau_estimate(groups):
    """Between-passage SD of HR (method of moments), pooled over condition cells with >= 4 passages."""
    taus = []
    for per in groups:
        rows = [(h / n, n) for h, n, _ in per.values() if n >= 5]
        if len(rows) < 4:
            continue
        p = np.array([r[0] for r in rows])
        m = np.array([r[1] for r in rows])
        pbar = (p * m).sum() / m.sum()
        within = np.mean(pbar * (1 - pbar) / m)
        taus.append(max(0.0, p.var(ddof=1) - within))
    return float(math.sqrt(np.mean(taus))) if taus else None


def auroc(scores, y):
    s = np.asarray(scores, float)
    y = np.asarray(y, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # tie-aware Mann-Whitney
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def threshold_at_fpr(scores, y, fpr=0.2):
    s = np.asarray(scores, float)
    neg = np.sort(s[np.asarray(y) == 0])
    if len(neg) == 0:
        return None
    return float(neg[int(math.floor((1 - fpr) * (len(neg) - 1)))])


def detector_table(con, runs):
    """Claim-level rows with final label and detector scores."""
    rows = []
    for r in runs:
        if r["judge_status"] != "done" or r["block"] == "v2gen":
            continue
        labs = labels_for(r["run_id"])
        if not labs:
            continue
        d = os.path.join(RUNS, r["run_id"])
        v5 = (jload(os.path.join(d, "v5.json"), {}) or {}).get("scores", {})
        v2 = (jload(os.path.join(d, "v2.json"), {}) or {}).get("scores", {})
        try:
            v3 = v3_scores(r["run_id"])
        except Exception:
            v3 = {}
        for l in labs:
            if l.get("seeded") or not l.get("checkable"):
                continue
            rows.append({"run": r["run_id"], "gen": r["gen"], "lang": r["lang"], "src": l["source"], "y": int(l["hallucinated"]),
                         "V5": v5.get(l["id"]), "V3": (-v3[l["id"]]["mean_lp"]) if l["id"] in v3 else None,
                         "V2": (v2.get(l["id"]) or {}).get("score")})
    return rows


def detector_stats(rows):
    out = {}
    for det in ("V3", "V5", "V2"):
        sub = [r for r in rows if r[det] is not None]
        y = [r["y"] for r in sub]
        s = [r[det] for r in sub]
        a = auroc(s, y)
        out[det] = {"n": len(sub), "positives": int(sum(y)), "auroc": a}
        for lang in ("greek", "hebrew"):
            ss = [r for r in sub if r["lang"] == lang]
            out[det]["auroc_" + lang] = auroc([r[det] for r in ss], [r["y"] for r in ss])
    # fusion on claims with all available detectors (V3, V5 always; V2 when present)
    fus = {}
    for dets in (("V3", "V5"), ("V3", "V5", "V2")):
        sub = [r for r in rows if all(r[d] is not None for d in dets)]
        y = np.array([r["y"] for r in sub], int)
        if len(sub) < 30 or y.sum() < 5 or (1 - y).sum() < 5:
            continue
        bins = []
        for d in dets:
            s = np.array([r[d] for r in sub], float)
            thr = threshold_at_fpr(s, y, 0.2)
            bins.append(s > thr)
        bins = np.array(bins)
        res = {"n": len(sub), "positives": int(y.sum()), "thresholds": "each detector at FPR~0.20"}
        for name, f in (("OR", lambda b: b.any(0)), ("AND", lambda b: b.all(0))) + (
                (("MAJ2of3", lambda b: b.sum(0) >= 2),) if len(dets) == 3 else ()):
            fused = f(bins)
            res[name] = {"TPR": float(fused[y == 1].mean()), "FPR": float(fused[y == 0].mean())}
        # independence forecast and pairwise shared-miss excess
        for cls, key in ((1, "TPR"), (0, "FPR")):
            pm = bins[:, y == cls].mean(1)
            res["indep_OR_" + key] = float(1 - np.prod(1 - pm))
            res["indep_AND_" + key] = float(np.prod(pm))
            if BRB is not None:
                try:
                    k = len(dets)
                    P2 = np.zeros((k, k))
                    for i in range(k):
                        for j in range(k):
                            P2[i, j] = float((bins[i, y == cls] & bins[j, y == cls]).mean())
                    q = BRB.maxent_joint(pm, P2)
                    A = BRB.atoms(k)
                    res["maxent_OR_" + key] = float(q[A.any(1)].sum())
                    res["maxent_AND_" + key] = float(q[A.all(1)].sum())
                    lo, hi = BRB.rule_rate_bounds(pm, P2, A.any(1).astype(int))
                    res["LP_OR_" + key] = [lo, hi]
                except Exception as e:
                    res["brb_error"] = repr(e)[:200]
        if BRB is not None:
            res["delta_plus"] = BRB.shared_miss_excess(list(bins), y, 1).round(4).tolist()
            res["delta_minus"] = BRB.shared_miss_excess(list(bins), y, 0).round(4).tolist()
        fus["+".join(dets)] = res
    return out, fus


def judge_quality(runs):
    seeded = caught = 0
    audit = agree = 0
    v1c_judgeC = v1c = v1s_judgeS = v1s = 0
    cites = fails = 0
    for r in runs:
        labs = labels_for(r["run_id"]) if r["judge_status"] == "done" else None
        if not labs:
            continue
        lab_meta = jload(os.path.join(RUNS, r["run_id"], "labels.json"))
        cites += lab_meta.get("citation_checked", 0)
        fails += lab_meta.get("citation_failed", 0)
        for l in labs:
            if l.get("seeded") and l.get("judge"):
                seeded += 1
                caught += l["judge"] in ("C", "U")
            if l.get("v1") in ("S", "C") and l.get("judge"):
                audit += 1
                jh = l["judge"] in ("C", "U")
                agree += (jh == (l["v1"] == "C"))
                if l["v1"] == "C":
                    v1c += 1
                    v1c_judgeC += jh
                else:
                    v1s += 1
                    v1s_judgeS += (l["judge"] == "S")
    return {"seeded": seeded, "seeded_caught": caught, "seeded_detection": (caught / seeded) if seeded else None,
            "audit_n": audit, "audit_agreement": (agree / audit) if audit else None,
            "judge_sensitivity_vs_v1": (v1c_judgeC / v1c) if v1c else None,
            "judge_specificity_vs_v1": (v1s_judgeS / v1s) if v1s else None,
            "citations_checked": cites, "citation_fail_rate": (fails / cites) if cites else None}


def _norm_txt(s):
    import re
    import unicodedata
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def retention(runs):
    """True-claim retention for the repair loop (plan D4 floor 0.70).
    Presence proxy: a Supported claim of the original is retained if its original-language text (orig) or, failing
    that, the first 40 characters of its span still appear in the delivered answer (normalized). Count proxy:
    Supported claims in delivered / Supported claims in original."""
    out = []
    for r in runs:
        if r["repair"] != "delivered" or r["judge_status"] != "done":
            continue
        o = r["parent"]
        lo, ld = labels_for(o), labels_for(r["run_id"])
        if not lo or not ld:
            continue
        co = {c["id"]: c for c in jload(os.path.join(RUNS, o, "claims.json"))["claims"]}
        with open(os.path.join(RUNS, r["run_id"], "answer.md"), encoding="utf-8") as f:
            dtxt = _norm_txt(f.read())
        s_o = [l for l in lo if l.get("final") == "S" and not l.get("seeded")]
        kept = 0
        for l in s_o:
            c = co.get(l["id"], {})
            key = _norm_txt(c.get("orig") or "") or _norm_txt((c.get("span") or "")[:40])
            if key and key in dtxt:
                kept += 1
        s_d = sum(1 for l in ld if l.get("final") == "S" and not l.get("seeded"))
        h_o = [l for l in lo if l.get("checkable") and not l.get("seeded")]
        h_d = [l for l in ld if l.get("checkable") and not l.get("seeded")]
        out.append({"run": r["run_id"], "lang": r["lang"], "grounding": r["grounding"], "S_orig": len(s_o),
                    "S_kept_presence": kept, "S_delivered": s_d,
                    "hr_orig": (sum(l["hallucinated"] for l in h_o) / len(h_o)) if h_o else None,
                    "hr_delivered": (sum(l["hallucinated"] for l in h_d) / len(h_d)) if h_d else None})
    if not out:
        return None
    tot_s = sum(x["S_orig"] for x in out)
    return {"answers": len(out),
            "retention_presence": (sum(x["S_kept_presence"] for x in out) / tot_s) if tot_s else None,
            "retention_count": (sum(x["S_delivered"] for x in out) / tot_s) if tot_s else None,
            "rows": out}


def stack_pairs(runs):
    """Paired MLX - GGUF raw HR on matched capped Qwen3.6 cells, passage-cluster bootstrap (H9; reported only)."""
    by = {r["run_id"]: r for r in runs}
    d, ps = [], []
    for r in runs:
        if r["block"] != "bridge" or r["judge_status"] != "done":
            continue
        g = r["run_id"].replace("bridge__qwen36_mlx__", "core__qwen36_35b__")
        if g not in by or by[g]["judge_status"] != "done":
            continue
        a = [l for l in labels_for(r["run_id"]) if l.get("checkable") and not l.get("seeded")]
        b = [l for l in labels_for(g) if l.get("checkable") and not l.get("seeded")]
        if a and b:
            d.append(sum(l["hallucinated"] for l in a) / len(a) - sum(l["hallucinated"] for l in b) / len(b))
            ps.append(r["passage"])
    if len(d) < 4:
        return None
    d = np.array(d)
    up = sorted(set(ps))
    idx = {p: [i for i, q in enumerate(ps) if q == p] for p in up}
    bs = [np.mean(np.concatenate([d[idx[up[j]]] for j in RNG.integers(0, len(up), len(up))])) for _ in range(4000)]
    return {"pairs": len(d), "mean_diff": float(d.mean()), "ci95": np.percentile(bs, [2.5, 97.5]).tolist(),
            "ci90_tost": np.percentile(bs, [5, 95]).tolist(), "equivalent_within_3pts": bool(
                np.percentile(bs, 5) > -0.03 and np.percentile(bs, 95) < 0.03),
            "note": "one draw per cell; engine confounded with quantization, prompt wrapper and seed"}


def progress(con, runs):
    pr = defaultdict(lambda: defaultdict(int))
    for r in runs:
        pr[r["block"]]["total"] += 1
        pr[r["block"]]["gen_" + r["gen_status"]] += 1
        pr[r["block"]]["labeled"] += r["judge_status"] == "done"
    # throughput by generator
    walls = defaultdict(list)
    for r in runs:
        m = jload(os.path.join(RUNS, r["run_id"], "meta.json"))
        if m and m.get("wall_s"):
            walls[(r["gen"], r["reasoning"])].append(m["wall_s"] + (m.get("load_s") or 0))
    eta = 0.0
    for r in runs:
        if r["gen_status"] == "pending" and r["block"] != "v2gen":
            w = walls.get((r["gen"], r["reasoning"]))
            eta += (np.mean(w) if w else {"qwen35_122b": 5400}.get(r["gen"], 600))
    return {k: dict(v) for k, v in pr.items()}, {"%s/%s" % k: round(float(np.mean(v)) / 60, 1) for k, v in walls.items()}, eta / 3600


def build(final=False):
    con = db.connect()
    runs = load_runs(con)
    ps = {p["id"]: p for p in jload(os.path.join(STATE, "passages.json"))}
    prog, minutes, eta_h = progress(con, runs)
    rep = {"generated": time.strftime("%Y-%m-%d %H:%M"), "spent": round(db.spent(con), 2), "cap": 75.0,
           "progress": prog, "minutes_per_run": minutes, "primary_gen_hours_left": round(eta_h, 1)}
    # raw HR by cell (core + big + bridge), delivered HR (repair)
    cells = defaultdict(list)
    for r in runs:
        if r["judge_status"] != "done":
            continue
        if r["block"] in ("core", "big", "bridge"):
            cells[(r["gen"], r["lang"], r["grounding"], r["reasoning"])].append((r["run_id"], r["passage"]))
        elif r["block"] == "repair":
            cells[("repair:" + (r["repair"] or ""), r["lang"], r["grounding"], r["reasoning"])].append((r["run_id"], r["passage"]))
    hr = {}
    per_groups = []
    for k, ids in sorted(cells.items()):
        per = cell_counts(ids)
        s = hr_summary(per)
        if s:
            hr["|".join(k)] = s
            if not k[0].startswith("repair"):
                per_groups.append(per)
    rep["raw_hr_cells"] = hr
    # marginal summaries
    marg = {}
    for dim, idx in (("generator", 0), ("language", 1), ("grounding", 2), ("reasoning", 3)):
        agg = defaultdict(list)
        for k, ids in cells.items():
            if k[0].startswith("repair"):
                continue
            agg[k[idx]].extend(ids)
        marg[dim] = {lvl: hr_summary(cell_counts(ids)) for lvl, ids in agg.items()}
    rep["raw_hr_marginal"] = marg
    rep["tau"] = tau_estimate(per_groups)
    # completeness and discard by claim type
    by_type = defaultdict(lambda: [0, 0, 0])
    sections = []
    for r in runs:
        if r["judge_status"] != "done":
            continue
        for l in labels_for(r["run_id"]) or []:
            if l.get("seeded"):
                continue
            t = by_type[l.get("type")]
            t[0] += l["final"] in ("C", "U")
            t[1] += l["final"] in ("S", "C", "U")
            t[2] += l["final"] == "N"
        cl = jload(os.path.join(RUNS, r["run_id"], "claims.json"), {})
        sections.append(len(set(cl.get("sections_answered") or [])) == 8)
    rep["by_claim_type"] = {k: {"hr": (v[0] / v[1]) if v[1] else None, "checkable": v[1],
                                "discard_rate": (v[2] / (v[1] + v[2])) if (v[1] + v[2]) else None} for k, v in by_type.items()}
    rep["completeness_all8"] = (sum(sections) / len(sections)) if sections else None
    rows = detector_table(con, runs)
    rep["detectors"], rep["fusion"] = detector_stats(rows)
    rep["judge_quality"] = judge_quality(runs)
    rep["v1_share_of_checkable"] = (sum(1 for r in rows if r["src"] == "v1") / len(rows)) if rows else None
    rep["repair_retention"] = retention(runs)
    rep["stack_bridge"] = stack_pairs(runs)
    jdump(rep, os.path.join(STATE, "report_%s.json" % time.strftime("%Y%m%d")))
    return rep


def pct(x):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else "%.0f%%" % (100 * x)


def summary_text(rep):
    L = []
    pr = rep["progress"]
    core = pr.get("core", {})
    gen_done = sum(v.get("gen_done", 0) for k, v in pr.items() if k != "v2gen")
    gen_tot = sum(v.get("total", 0) for k, v in pr.items() if k != "v2gen")
    lab = sum(v.get("labeled", 0) for k, v in pr.items() if k != "v2gen")
    L.append("Generated %d/%d primary answers (core %d/%d), %d labeled; API spend $%.2f of $%.0f; ~%.0f h of primary "
             "generation left." % (gen_done, gen_tot, core.get("gen_done", 0), core.get("total", 0), lab, rep["spent"],
                                   rep["cap"], rep["primary_gen_hours_left"]))
    m = rep["raw_hr_marginal"]
    for dim in ("generator", "grounding", "language", "reasoning"):
        parts = []
        for lvl, s in sorted((m.get(dim) or {}).items()):
            if s:
                parts.append("%s %s (n=%d claims, %d passages)" % (GEN_LABEL.get(lvl, lvl), pct(s["hr"]), s["checkable"], s["passages"]))
        if parts:
            L.append("Raw HR by %s: %s." % (dim, "; ".join(parts)))
    jq = rep["judge_quality"]
    if jq.get("citations_checked"):
        L.append("Judge: citation failures %s; seeded claims caught %s of %d; agreement with V1 on audit sample %s (n=%d)." % (
            pct(jq["citation_fail_rate"]), pct(jq["seeded_detection"]), jq["seeded"], pct(jq["audit_agreement"]), jq["audit_n"]))
    det = rep["detectors"]
    dparts = ["%s AUROC %.2f (n=%d)" % (k, v["auroc"], v["n"]) for k, v in det.items() if v.get("auroc") is not None]
    if dparts:
        L.append("Detectors: " + "; ".join(dparts) + ".")
    rt = rep.get("repair_retention")
    if rt:
        L.append("Repair: %d delivered answers; true-claim retention %s (text-presence proxy), %s (count proxy)." % (
            rt["answers"], pct(rt["retention_presence"]), pct(rt["retention_count"])))
    sb = rep.get("stack_bridge")
    if sb:
        L.append("Stack bridge (MLX-GGUF, %d pairs): %+.1f pts, 95%% CI %+.1f to %+.1f." % (
            sb["pairs"], 100 * sb["mean_diff"], 100 * sb["ci95"][0], 100 * sb["ci95"][1]))
    if rep.get("tau") is not None:
        L.append("Passage spread tau ≈ %.3f." % rep["tau"])
    return "\n".join(L)


if __name__ == "__main__":
    rep = build(final="--final" in sys.argv)
    print(summary_text(rep))
