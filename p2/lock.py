"""harness.lock: everything that can change an answer, pinned for the frozen harness (v1.0.0, 20 Oct 2026, Thor:
Qwen3.5-122B, web research off, V2 off, fusion "2 of 3" over V3 + V5 (both must flag), thresholds v1).

python -m p2.lock build   -> writes harness.lock at the repo root
python -m p2.lock verify  -> exits non-zero if any pinned code, setting, threshold or weight differs
"""
import hashlib
import json
import os
import subprocess
import sys

from . import harness
from .common import REPO, OPENDATA, STATE, jload

LOCK = os.path.join(REPO, "harness.lock")
VERSION = "1.0.1"
GEN = "qwen35_122b"
V5_GEN = "gemma4_26b"
CODE = ["p2/harness.py", "p2/prompts.py", "p2/texts.py", "p2/morph.py", "p2/witnesses.py", "p2/extract.py",
        "p2/v1.py", "p2/judge.py", "p2/detectors.py", "p2/fusion.py", "p2/repair.py", "p2/stagec.py",
        "p2/staged.py", "p2/claude_cli.py", "p2/labelcli.py"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def gguf_shards(first):
    d, b = os.path.split(first)
    if "-00001-of-" in b:
        stem, tail = b.split("-00001-of-")
        n = int(tail.split(".")[0])
        return [os.path.join(d, "%s-%05d-of-%s" % (stem, i, tail)) for i in range(1, n + 1)]
    return [first]


def snapshot():
    from .stagec import CTX
    from . import claude_cli, extract, judge
    th = jload(os.path.join(STATE, "thresholds_v1.json"))
    cfg = harness.GENERATORS[GEN]
    od = {}
    for name in sorted(os.listdir(OPENDATA)):
        p = os.path.join(OPENDATA, name)
        if os.path.isdir(os.path.join(p, ".git")):
            od[name] = subprocess.run(["git", "-C", p, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    claude_v = subprocess.run([claude_cli.CLAUDE, "--version"], capture_output=True, text=True).stdout.strip()
    return {
        "version": VERSION,
        "generator": {"key": GEN, "label": cfg["label"], "weights": {os.path.basename(s): sha(s) for s in gguf_shards(cfg["gguf"])},
                      "engine": harness.llama_version(),
                      "server": "ultra-llm-svc server-v1.1 via gateway :8100 (generation); %s under a gateway lease (V5)"
                                % harness.LLAMA_SERVER,
                      "server_args": cfg["args"] + CTX + ["--jinja", "--reasoning-format", "none", "-np", "1"],
                      "metal_tensor_api": "default (on); GGML_METAL_TENSOR_DISABLE unset",
                      "sampling_capped": cfg["sampling"]["capped"], "think_cap": harness.THINK_CAP,
                      "answer_max": harness.ANSWER_MAX, "force_close": harness.FORCE_CLOSE},
        "grounding": "G3", "web_research": "off", "reasoning": "capped",
        "detectors": {"V1": "source and morphology check (p2/v1.py)", "V3": "generator token log-probabilities",
                      "V5": {"model": V5_GEN, "weights": sha(harness.GENERATORS[V5_GEN]["gguf"])}, "V2": "off"},
        "fusion": {"rule": "2of3", "detectors": ["V3", "V5"], "effective": "both V3 and V5 must flag (V1 contradictions always flagged)",
                   "thresholds": {"V3": th["V3"][GEN], "V5": th["V5"][GEN]}, "fpr_target": th["fpr_target"]},
        "repair": "one pass on the cell's own G3 packet; re-check with the same rule; flagged sentences cut",
        "extractor": {"model": extract.EXTRACT_MODEL, "version": extract.EXTRACT_VERSION},
        "judge": {"model": judge.JUDGE_MODEL, "rubric": judge.RUBRIC_VERSION, "effort": claude_cli.EFFORT,
                  "via": "Claude Code CLI on the Max plan", "cli": claude_v},
        "opendata": od,
        "code": {c: sha(os.path.join(REPO, c)) for c in CODE},
    }


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    snap = snapshot()
    if cmd == "build":
        with open(LOCK, "w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False, indent=1)
        print("wrote", LOCK, sha(LOCK)[:16])
        return
    lock = json.load(open(LOCK, encoding="utf-8"))
    diffs = []

    def walk(a, b, path=""):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in set(a) | set(b):
                walk(a.get(k), b.get(k), path + "/" + k)
        elif a != b:
            diffs.append(path)
    walk(lock, snap)
    diffs = [d for d in diffs if not d.startswith("/judge/cli")]  # CLI auto-updates; model and rubric are what count
    if diffs:
        print("LOCK MISMATCH:", diffs)
        sys.exit(1)
    print("lock ok", sha(LOCK)[:16])


if __name__ == "__main__":
    main()
