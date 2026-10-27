"""Stage C pre-flight checks: seed fix, cut(), web research + notes, judge packet web block, flags with v0 thresholds."""
import json, os, sys, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from p2.judge import seeded_claims
from p2.localq import passages, corpus
from p2.common import RUNS, STATE, jload
from p2 import web, fusion
from p2.stagec import cut

ps, c = passages(), corpus()
ok = True

# 1) seed fix: the defective Mark 10:5 seed is gone; count seeds that changed across all judged runs
s = seeded_claims("stageB__qwen35_122b__Mark.9.14-9.29__G2__capped", ps["Mark.9.14-9.29"], c)
print("seed kinds now:", [x["seed_kind"] for x in s], [x.get("ref") for x in s])
ok &= not any(x["seed_kind"] == "swapref" and x["ref"] == "Mark 10:5" for x in s)
changed = total = 0
for d in glob.glob(os.path.join(RUNS, "*", "labels.json")):
    rid = d.split("/")[-2]
    lab = jload(d)["labels"]
    old = sum(1 for l in lab if l.get("seeded"))
    if not old:
        continue
    meta = jload(os.path.join(RUNS, rid, "meta.json"), {})
    pid = meta.get("passage")
    if pid not in ps:
        continue
    total += 1
    changed += len(seeded_claims(rid, ps[pid], c)) != old
print("runs with seeds:", total, "seed count changed by the fix:", changed)

# 2) cut()
ans = "First claim is fine. Second claim is wrong; it says X. Third is fine.\nFourth line."
cl = [{"id": "c1", "span": "Second claim is wrong", "orig": None}, {"id": "c2", "span": "Third is fine", "orig": None}]
new, ids = cut(ans, cl, {"c1"})
print("cut:", repr(new), ids)
ok &= "Second" not in new and "Third is fine" in new and ids == ["c1"]

# 3) web research on one calibration passage
rid = "stageC__TEST__web"
os.makedirs(os.path.join(RUNS, rid), exist_ok=True)
rec = web.research(rid, ["John 2:1-11 wedding at Cana water into wine commentary",
                         "John 2:4 τί ἐμοὶ καὶ σοί γύναι interpretation"])
print("pages:", [(p["n"], p["domain"], p["chars"]) for p in rec["pages"]])
nb = web.notes_block(rid)
print("notes chars:", len(nb))
ok &= len(rec["pages"]) >= 2 and "[X:1]" in nb

# 4) thresholds + flags on a Stage B run with V3 only (no V5) as a smoke check
th = jload(fusion.TH_V0)
print("thresholds:", json.dumps({k: th[k] for k in ("V3", "V5", "V2", "n")}))
fl = fusion.flags("stageB__qwen35_122b__Mark.9.14-9.29__G3__capped", "qwen35_122b", th, "any", ["V3", "V5"])
print("flags (V3-only available):", len(fl), fl[:3])
print("ALL_OK" if ok else "FAIL")
