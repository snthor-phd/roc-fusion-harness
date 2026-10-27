"""SQLite state for the Phase 0 queue (shared by the local-compute and API daemons)."""
import json
import os
import sqlite3
import time

from .common import STATE

DB = os.path.join(STATE, "p2.db")

STAGES = ["gen", "extract", "v1", "judge", "v5", "repair", "v2"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, block TEXT, gen TEXT, passage TEXT, lang TEXT, grounding TEXT, reasoning TEXT,
  repair TEXT, rep INTEGER, seed INTEGER, ord INTEGER, parent TEXT,
  gen_status TEXT DEFAULT 'pending', extract_status TEXT DEFAULT 'pending', v1_status TEXT DEFAULT 'pending',
  judge_status TEXT DEFAULT 'pending', v5_status TEXT DEFAULT 'pending', repair_status TEXT DEFAULT 'na',
  v2_status TEXT DEFAULT 'na', attempts INTEGER DEFAULT 0, error TEXT, updated REAL
);
CREATE TABLE IF NOT EXISTS batches (
  batch_id TEXT PRIMARY KEY, kind TEXT, model TEXT, n INTEGER, reserved REAL, status TEXT, submitted REAL,
  ended REAL, custom_ids TEXT
);
CREATE TABLE IF NOT EXISTS ledger (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, kind TEXT, model TEXT, custom_id TEXT, input_tokens INTEGER,
  cache_write INTEGER, cache_read INTEGER, output_tokens INTEGER, cost REAL, note TEXT
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, who TEXT, msg TEXT
);
"""


def connect():
    con = sqlite3.connect(DB, timeout=60, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=60000")
    con.executescript(SCHEMA)
    return con


def event(con, who, msg):
    con.execute("INSERT INTO events(ts, who, msg) VALUES (?,?,?)", (time.time(), who, msg[:2000]))


def set_status(con, run_id, stage, status, error=None):
    con.execute("UPDATE runs SET %s_status=?, error=COALESCE(?, error), updated=? WHERE run_id=?" % stage,
                (status, error, time.time(), run_id))


def insert_runs(con, runs):
    n = 0
    for r in runs:
        cur = con.execute("SELECT 1 FROM runs WHERE run_id=?", (r["run_id"],)).fetchone()
        if cur:
            continue
        no_label = r["block"] == "v2gen"
        con.execute("""INSERT INTO runs(run_id, block, gen, passage, lang, grounding, reasoning, repair, rep, seed, ord,
                       parent, repair_status, v2_status, extract_status, v1_status, judge_status, v5_status, updated)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (r["run_id"], r["block"], r["gen"], r["passage"], r["lang"], r["grounding"], r["reasoning"],
                     r.get("repair"), r.get("rep", 1), r["seed"], r["order"], r.get("parent"),
                     "pending" if r.get("repair") == "on" else "na",
                     "pending" if (r["block"] == "core" and r["gen"] == "qwen38_27b") else "na",
                     "na" if no_label else "pending", "na" if no_label else "pending",
                     "na" if no_label else "pending", "na" if no_label else "pending", time.time()))
        n += 1
    return n


def spent(con):
    row = con.execute("SELECT COALESCE(SUM(cost),0) FROM ledger").fetchone()
    return row[0]


def reserved(con):
    row = con.execute("SELECT COALESCE(SUM(reserved),0) FROM batches WHERE status IN ('submitted','in_progress')").fetchone()
    return row[0]
