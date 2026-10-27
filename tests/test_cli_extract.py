"""Smoke test: extract one Phase 0 answer through the Claude CLI on Opus 5.5; compare with Phase 0's Opus 4.8 extraction."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from p2 import extract, claude_cli
from p2.common import RUNS, jload
from p2.localq import passages

rid = sys.argv[1] if len(sys.argv) > 1 else "core__qwen38_27b__1Cor.6.1-6.8__G2__capped"
d = os.path.join(RUNS, rid)
p = passages()[rid.split("__")[2]]
answer = open(os.path.join(d, "answer.md"), encoding="utf-8").read()
req = extract.build_request({"run_id": rid}, p, answer)
out, meta = claude_cli.run(req)
old = jload(os.path.join(d, "claims.json"), {})
res = {"run_id": rid, "meta": meta, "n_claims_opus55": len(out.get("claims", [])),
       "n_claims_opus48_phase0": len(old.get("claims", [])), "sections": out.get("sections"),
       "first_rows": out.get("claims", [])[:3]}
json.dump(out, open(os.path.join(__import__("tempfile").gettempdir(), "cli_extract_test.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1)[:2500])
