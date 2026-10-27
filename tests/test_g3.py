"""Stage B checks: G2 prompts byte-identical to Phase 0; G3 packets build and carry the expected tables."""
import glob, os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from p2.common import RUNS
from p2.localq import passages, corpus
from p2.prompts import generation_prompt, grounding_packet

ps, c = passages(), corpus()
# 1) G2 regression against every Phase 0 core G2 prompt
same = diff = 0
for d in sorted(glob.glob(os.path.join(RUNS, "core__*__G2__*"))):
    if "__v2s" in d or d.endswith("__delivered"):
        continue
    pid = os.path.basename(d).split("__")[2]
    old = open(os.path.join(d, "prompt.txt"), encoding="utf-8").read()
    if generation_prompt(ps[pid], c, "G2") == old:
        same += 1
    else:
        diff += 1
print("G2 regression: %d identical, %d different" % (same, diff))

# 2) G3 for the 8 shared (half) design passages
half = [p for p in ps.values() if p["split"] == "design" and p.get("half")]
out = []
for p in sorted(half, key=lambda p: (p["lang"], p["id"])):
    g2 = grounding_packet(p, c, "G2")
    g3 = grounding_packet(p, c, "G3")
    pr = generation_prompt(p, c, "G3")
    chk = {
        "starts_with_g2": g3.startswith(g2),
        "occurrence": "[O:" in g3,
        "witness": "[W:" in g3 if p["lang"] == "greek" else "not available for the Hebrew Bible" in g3,
        "lxx": "[L:" in g3 if p["lang"] == "hebrew" else "not lemmatized" in g3,
        "g3_wording": "manuscript contents, word occurrence counts" in pr,
    }
    out.append({"passage": p["id"], "lang": p["lang"], "g2_chars": len(g2), "g3_chars": len(g3),
                "prompt_chars": len(pr), "checks_pass": all(chk.values()), **chk})
for o in out:
    print(json.dumps(o))
occ = [l for l in grounding_packet(half[0], c, "G3").splitlines() if l.startswith("[O:")]
print("sample occurrence rows:", occ[:3])
ok = diff == 0 and same > 0 and all(o["checks_pass"] for o in out) and len(out) == 8
print("ALL_OK" if ok else "FAIL")
