"""Phase 0 design matrix (plan Section 8, Phase 0) with committed, seeded run orders."""
import random

from .sampler import SEED

CORE_GENS = ["qwen38_27b", "qwen36_35b", "gemma4_26b"]
GROUNDING = ["G0", "G2"]
REASONING = ["off", "capped"]
V2_N = 5  # resamples per 27B core cell (V2), lowest priority

# local-queue priorities (lower runs first)
PRI = {"core": 10, "repair_chain": 15, "bridge": 20, "repair": 30, "v5": 40, "big": 50, "v5_late": 60,
       "v2gen": 80, "v2score": 75}
# labeling (API) priority order per D22 decision: core, then bridge, then repair; 122B last (not named in D22)
LABEL_PRI = {"core": 1, "bridge": 2, "repair": 3, "big": 4}


def phase0_runs(passages):
    design = [p for p in passages if p["split"] == "design"]
    half = [p for p in design if p.get("half")]
    runs = []

    def add(block, gen, p, g, r, extra=None):
        rid = "%s__%s__%s__%s__%s" % (block, gen, p["id"], g, r)
        if extra:
            rid += "__" + extra
        d = {"run_id": rid, "block": block, "gen": gen, "passage": p["id"], "lang": p["lang"], "grounding": g,
             "reasoning": r, "repair": None, "rep": 1}
        runs.append(d)
        return d

    for p in design:
        for gen in CORE_GENS:
            for g in GROUNDING:
                for r in REASONING:
                    add("core", gen, p, g, r)
    for p in half:
        for g in GROUNDING:
            add("big", "qwen35_122b", p, g, "capped")
    for p in design:
        for g in GROUNDING:
            add("bridge", "qwen36_mlx", p, g, "capped")
    for p in half:
        for g in GROUNDING:
            for rep in ("off", "on"):
                d = add("repair", "qwen38_27b", p, g, "off", "repair-" + rep)
                d["repair"] = rep
    # seeded randomized order within each block (passages are blocks; order randomized across the block's cells)
    rng = random.Random(SEED + 1)
    out = []
    for block in ("core", "bridge", "repair", "big"):
        rs = sorted([r for r in runs if r["block"] == block], key=lambda r: r["run_id"])
        rng.shuffle(rs)
        for i, r in enumerate(rs):
            r["order"] = i
            r["seed"] = rng.randrange(1, 2 ** 31)
        out.extend(rs)
    return out


def v2_runs(runs):
    """V2 resamples for the 27B core cells: all reasoning-off cells get their N resamples first."""
    rng = random.Random(SEED + 2)
    out = []
    base = [r for r in runs if r["block"] == "core" and r["gen"] == "qwen38_27b"]
    base.sort(key=lambda r: (r["reasoning"] != "off", r["order"]))
    for k in range(1, V2_N + 1):
        for r in base:
            d = dict(r)
            d["run_id"] = r["run_id"] + "__v2s%d" % k
            d["block"] = "v2gen"
            d["parent"] = r["run_id"]
            d["rep"] = k + 1
            d["seed"] = rng.randrange(1, 2 ** 31)
            d["k"] = k
            out.append(d)
    out.sort(key=lambda d: (d["reasoning"] != "off", d["k"], d["order"]))
    for i, d in enumerate(out):
        d["order"] = i
    return out
