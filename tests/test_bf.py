"""Smoke test for budget forcing on the 122B: short task, MIN_THINK 1500, cap 4000, answer 400."""
import os, sys, json
os.environ.update(P2_MIN_THINK="1500", P2_THINK_CAP="4000", P2_ANSWER_MAX="400")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from p2 import harness
srv = harness.LlamaServer("qwen35_122b", extra_args=["-c", "16384"])
srv.start()
try:
    ans, think, lps, meta = harness.generate_llama(srv, "qwen35_122b", "What is 17 * 23? Answer in one line.", "capped", 4242)
finally:
    srv.stop()
print(json.dumps({"forces": meta.get("budget_forces"), "think_tokens": meta.get("think_tokens"),
                  "closed": meta.get("think_closed_naturally"), "waits_in_thinking": think.count("Wait,"),
                  "answer": ans.strip()[:200], "answer_tokens": meta.get("answer_tokens")}, indent=1))
