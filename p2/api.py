"""Anthropic Message Batches client with a hard spending cap enforced in code (plan 12.1, D22).

Cap rule: a batch is submitted only if  spent + outstanding reservations + this batch's worst case <= CAP.
Worst case per request = estimated input tokens at the uncached batch input price + max_tokens at the batch output
price. Actual cost is booked from the usage block of each result with conservative cache prices.
"""
import json
import os
import time
import urllib.error
import urllib.request

from . import db
from .common import P2, STATE

CAP_USD = 75.00
KEY_FILE = os.path.join(P2, ".anthropic_key")
API = "https://api.anthropic.com/v1"
VERSION = "2023-06-01"

# USD per million tokens, Batch API (50% of standard). Cache prices set conservatively at the *standard* rates.
PRICES = {
    "claude-opus-5-5": {"in": 2.0, "out": 10.0, "cache_write": 5.0, "cache_read": 0.20},
    "claude-opus-4-8": {"in": 2.5, "out": 12.5, "cache_write": 6.25, "cache_read": 0.50},
}


def _key():
    with open(KEY_FILE) as f:
        return f.read().strip()


def _req(method, path, body=None, timeout=120, raw=False):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API + path if path.startswith("/") else path, data=data, method=method,
                                 headers={"x-api-key": _key(), "anthropic-version": VERSION,
                                          "content-type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                b = r.read()
                return b if raw else json.loads(b)
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:500]
            if e.code in (429, 500, 502, 503, 529) and attempt < 4:
                time.sleep(20 * (attempt + 1))
                continue
            raise RuntimeError("HTTP %s %s: %s" % (e.code, path, msg))
        except urllib.error.URLError:
            if attempt < 4:
                time.sleep(20 * (attempt + 1))
                continue
            raise


def est_tokens(params):
    """Conservative input-token estimate: UTF-8 bytes / 1.8 (calibrated 12 Oct with count_tokens: true counts were
    1.0-1.3x the bytes/2.5 estimate on Greek packets)."""
    n = len(json.dumps(params.get("system", ""), ensure_ascii=False).encode())
    n += len(json.dumps(params.get("messages", []), ensure_ascii=False).encode())
    n += len(json.dumps(params.get("tools", []), ensure_ascii=False).encode())
    return int(n / 1.8) + 50


def worst_case(model, params):
    p = PRICES[model]
    return (est_tokens(params) * p["in"] + params["max_tokens"] * p["out"]) / 1e6


def cost_of(model, usage):
    p = PRICES[model]
    return (usage.get("input_tokens", 0) * p["in"] + usage.get("cache_creation_input_tokens", 0) * p["cache_write"]
            + usage.get("cache_read_input_tokens", 0) * p["cache_read"] + usage.get("output_tokens", 0) * p["out"]) / 1e6


def headroom(con):
    return CAP_USD - db.spent(con) - db.reserved(con)


def submit(con, kind, model, items):
    """items: list of (custom_id, params). Submits the longest prefix that fits under the cap.
    Returns (batch_id or None, n_submitted)."""
    if os.path.exists(os.path.join(STATE, "API_STOP")):
        return None, 0
    room = headroom(con)
    chosen, total, idmap = [], 0.0, {}
    import hashlib
    for cid, params in items:
        params = dict(params)
        params["model"] = model
        w = worst_case(model, params)
        if total + w > room:
            break
        # API custom_id must match ^[a-zA-Z0-9_-]{1,64}$ ; keep a private map to the run id
        safe = kind[0] + "_" + hashlib.sha1(cid.encode()).hexdigest()[:24]
        idmap[safe] = cid
        chosen.append({"custom_id": safe, "params": params})
        total += w
    if not chosen:
        if items:
            db.event(con, "api", "cap reached: headroom $%.2f cannot fit next %s request" % (room, kind))
        return None, 0
    r = _req("POST", "/messages/batches", {"requests": chosen}, timeout=300)
    bid = r["id"]
    con.execute("INSERT INTO batches(batch_id, kind, model, n, reserved, status, submitted, custom_ids) VALUES (?,?,?,?,?,?,?,?)",
                (bid, kind, model, len(chosen), total, "submitted", time.time(), json.dumps(idmap)))
    db.event(con, "api", "submitted %s batch %s: %d requests, reserved $%.2f (spent $%.2f)" % (kind, bid, len(chosen), total, db.spent(con)))
    return bid, len(chosen)


def poll(con, handler):
    """Check open batches; for ended ones book costs and pass each result to handler(kind, custom_id, result)."""
    rows = con.execute("SELECT * FROM batches WHERE status IN ('submitted','in_progress')").fetchall()
    for b in rows:
        try:
            info = _req("GET", "/messages/batches/%s" % b["batch_id"])
        except Exception as e:
            db.event(con, "api", "poll error %s: %r" % (b["batch_id"], e))
            continue
        if info.get("processing_status") != "ended":
            if b["status"] != "in_progress":
                con.execute("UPDATE batches SET status='in_progress' WHERE batch_id=?", (b["batch_id"],))
            continue
        raw = _req("GET", info["results_url"], timeout=600, raw=True)
        idmap = json.loads(b["custom_ids"] or "{}")
        n_ok = 0
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            res = json.loads(line)
            cid = idmap.get(res["custom_id"], res["custom_id"])
            rr = res.get("result", {})
            if rr.get("type") == "succeeded":
                msg = rr["message"]
                c = cost_of(b["model"], msg.get("usage", {}))
                u = msg.get("usage", {})
                con.execute("""INSERT INTO ledger(ts, kind, model, custom_id, input_tokens, cache_write, cache_read,
                               output_tokens, cost, note) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                            (time.time(), b["kind"], b["model"], cid, u.get("input_tokens", 0),
                             u.get("cache_creation_input_tokens", 0), u.get("cache_read_input_tokens", 0),
                             u.get("output_tokens", 0), c, msg.get("model", "")))
                n_ok += 1
            try:
                handler(b["kind"], cid, rr)
            except Exception as e:
                db.event(con, "api", "handler error %s %s: %r" % (b["kind"], cid, e))
        con.execute("UPDATE batches SET status='ended', ended=?, reserved=0 WHERE batch_id=?", (time.time(), b["batch_id"]))
        db.event(con, "api", "batch %s ended: %d/%d succeeded; spent now $%.2f" % (b["batch_id"], n_ok, b["n"], db.spent(con)))


def tool_input(result):
    """Return the forced tool call's input from a succeeded batch result (or None)."""
    if result.get("type") != "succeeded":
        return None
    for blk in result["message"].get("content", []):
        if blk.get("type") == "tool_use":
            return blk.get("input")
    # fallback: a JSON object in the text
    import re as _re
    for blk in result["message"].get("content", []):
        if blk.get("type") == "text":
            m = _re.search(r"\{.*\}", blk.get("text", ""), _re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except ValueError:
                    pass
    return None
