"""Claude Code CLI adapter (harness plan v2): runs extraction and judging on the Max plan, model claude-opus-5-5.

Takes the same request dicts the Batch API path built (system, one user message, one forced tool with an
input_schema) and returns (tool_input_dict, meta). The tool schema becomes --json-schema structured output.
ANTHROPIC_API_KEY is removed from the environment so the call bills the plan, never the API.
"""
import json
import os
import shutil
import subprocess
import tempfile
import time

MODEL = os.environ.get("P2_CLAUDE_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("P2_CLAUDE_EFFORT", "medium")  # matches the API default effort used in Phase 0
CLAUDE = shutil.which("claude") or "/opt/homebrew/bin/claude"


def _system_text(req):
    sys_ = req.get("system", "")
    if isinstance(sys_, list):
        sys_ = "\n\n".join(b.get("text", "") for b in sys_ if isinstance(b, dict))
    return sys_


def _user_text(req):
    msgs = req.get("messages") or []
    assert len(msgs) == 1 and msgs[0]["role"] == "user", "one user message expected"
    c = msgs[0]["content"]
    if isinstance(c, list):
        c = "\n\n".join(b.get("text", "") for b in c if isinstance(b, dict))
    return c


def run(req, timeout=1800, model=MODEL, effort=EFFORT):
    tools = req.get("tools") or []
    assert len(tools) == 1, "one tool (the output schema) expected"
    schema = tools[0]["input_schema"]
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(_system_text(req) + "\n\nReturn your output as the structured JSON object only.")
        sys_path = f.name
    cmd = [CLAUDE, "-p", "--model", model, "--effort", effort, "--output-format", "json",
           "--system-prompt-file", sys_path, "--json-schema", json.dumps(schema), "--tools", "",
           "--strict-mcp-config", "--setting-sources", "", "--no-session-persistence",
           "--disable-slash-commands"]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, input=_user_text(req), capture_output=True, text=True, timeout=timeout,
                           env=env, cwd=tempfile.gettempdir())
    finally:
        os.unlink(sys_path)
    wall = round(time.time() - t0, 1)
    try:
        d = json.loads(p.stdout)
    except Exception:
        raise RuntimeError("claude CLI: non-JSON output (rc=%s): %s %s" % (p.returncode, p.stdout[-400:], p.stderr[-400:]))
    if d.get("is_error"):
        raise RuntimeError("claude CLI error: %s" % str(d.get("result"))[:400])
    out = d.get("structured_output")
    if out is None:
        txt = (d.get("result") or "").strip()
        if txt.startswith("```"):
            txt = txt.strip("`").split("\n", 1)[-1]
        out = json.loads(txt)
    usage = d.get("usage") or {}
    meta = {"via": "claude-cli", "model": model, "effort": effort, "models_used": list((d.get("modelUsage") or {}).keys()),
            "wall_s": wall, "num_turns": d.get("num_turns"), "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"), "cache_read": usage.get("cache_read_input_tokens"),
            "plan_equivalent_usd": d.get("total_cost_usd"), "ts": time.time()}
    if model not in meta["models_used"]:
        raise RuntimeError("claude CLI answered with %s, not %s" % (meta["models_used"], model))
    return out, meta
