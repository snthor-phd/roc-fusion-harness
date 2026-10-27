"""Web research for the harness (Stage C): the generator plans queries, the harness searches the local SearXNG,
fetches the top pages and builds a notes block. Every web fact must be cited as [X:n]; the judge gets the same
excerpts so those citations can be checked.

Files per run: web.json (queries, search results, fetched pages). The open web is allowed (plan v2, question 2);
every domain is logged.
"""
import json
import os
import re
import html
import subprocess
import time
import urllib.parse
import urllib.request

from .common import RUNS, P2, jload, jdump

SEARX = os.environ.get("P2_SEARX", "http://127.0.0.1:8888")
SEARX_DIR = os.path.join(os.path.dirname(P2), "searxng")
MAX_QUERIES = 4
RESULTS_PER_QUERY = 3
MAX_PAGES = 6
EXCERPT_CHARS = 2500
SKIP = re.compile(r"(youtube\.com|youtu\.be|tiktok\.com|facebook\.com|instagram\.com|\.pdf$)", re.I)
UA = "Mozilla/5.0 (Macintosh; Apple Silicon) program2-harness/0.3 (research)"

PLAN_SCHEMA = {"type": "object", "properties": {"queries": {"type": "array", "maxItems": MAX_QUERIES,
               "items": {"type": "string"}}}, "required": ["queries"]}

PLAN_PROMPT = """You are preparing to write the exegesis below. The SOURCE MATERIALS already give the text, apparatus, morphology, manuscript contents, word counts and Septuagint. Plan web research for what they do NOT cover: the scholarly positions you will cite (who argued what), the history of interpretation, cross-references and parallels, and textual evidence beyond the packet.

Write up to {n} short web search queries (each under 12 words, specific, in English, naming the passage where useful). Return JSON {{"queries": [...]}}.

TASK
{task}"""


def searx_up():
    try:
        urllib.request.urlopen(SEARX + "/healthz", timeout=5)
        return True
    except Exception:
        try:
            urllib.request.urlopen(SEARX + "/", timeout=5)
            return True
        except Exception:
            return False


def ensure_searx():
    if searx_up():
        return True
    env = dict(os.environ, SEARXNG_SETTINGS_PATH=os.path.join(SEARX_DIR, "settings_ultra.yml"))
    log = open(os.path.join(SEARX_DIR, "webapp.log"), "a")
    subprocess.Popen([os.path.join(SEARX_DIR, ".venv", "bin", "python"), "-m", "searx.webapp"], cwd=SEARX_DIR,
                     env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(30):
        time.sleep(2)
        if searx_up():
            return True
    return False


def search(query):
    q = urllib.parse.urlencode({"q": query, "format": "json"})
    d = json.load(urllib.request.urlopen("%s/search?%s" % (SEARX, q), timeout=40))
    return [{"title": r.get("title"), "url": r.get("url"), "snippet": (r.get("content") or "")[:300],
             "engine": r.get("engine")} for r in d.get("results", [])[:RESULTS_PER_QUERY + 3]]


def page_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    raw = urllib.request.urlopen(req, timeout=30).read(3_000_000)
    try:
        import trafilatura
        txt = trafilatura.extract(raw, include_comments=False, include_tables=True, favor_recall=True)
        if txt and len(txt) > 200:
            return re.sub(r"[ \t]+", " ", txt).strip()
    except Exception:
        pass
    s = raw.decode("utf-8", "replace")
    s = re.sub(r"(?is)<(script|style|nav|footer|header|aside).*?</\1>", " ", s)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"(?s)<[^>]+>", " ", s))).strip()


def _excerpt(text, terms):
    """The EXCERPT_CHARS window of the page with the most query-term hits (pages are often long)."""
    if len(text) <= EXCERPT_CHARS:
        return text
    low = text.lower()
    best, best_i = -1, 0
    for i in range(0, len(text) - EXCERPT_CHARS + 1, 500):
        w = low[i:i + EXCERPT_CHARS]
        sc = sum(w.count(t) for t in terms)
        if sc > best:
            best, best_i = sc, i
    return text[best_i:best_i + EXCERPT_CHARS]


def research(run_id, queries):
    """Search and fetch; writes web.json; returns the record."""
    ensure_searx()
    rec = {"run_id": run_id, "ts": time.time(), "queries": queries, "searches": [], "pages": []}
    seen = set()
    terms = sorted({t.lower() for q in queries for t in re.findall(r"[A-Za-zͰ-Ͽἀ-῿֐-׿]{4,}", q)})
    for q in queries[:MAX_QUERIES]:
        try:
            res = search(q)
        except Exception as e:
            rec["searches"].append({"query": q, "error": repr(e)[:200]})
            continue
        rec["searches"].append({"query": q, "results": res})
        for r in res[:RESULTS_PER_QUERY]:
            u = r.get("url")
            if not u or u in seen or len(rec["pages"]) >= MAX_PAGES or SKIP.search(u):
                continue
            seen.add(u)
            try:
                txt = page_text(u)
                ok = len(txt) > 500
            except Exception as e:
                txt, ok = "", False
                r["fetch_error"] = repr(e)[:200]
            if ok:
                rec["pages"].append({"n": len(rec["pages"]) + 1, "url": u, "title": r.get("title"),
                                     "domain": urllib.parse.urlparse(u).netloc, "query": q,
                                     "excerpt": _excerpt(txt, terms), "chars": len(txt)})
    jdump(rec, os.path.join(RUNS, run_id, "web.json"))
    return rec


def notes_block(run_id):
    rec = jload(os.path.join(RUNS, run_id, "web.json"))
    if not rec or not rec.get("pages"):
        return ""
    lines = ["WEB RESEARCH (fetched %s; less authoritative than the source materials; cite any fact you take from "
             "it with its tag, e.g. [X:1]):" % time.strftime("%Y-%m-%d", time.localtime(rec["ts"]))]
    for pg in rec["pages"]:
        lines.append("[X:%d] %s — %s\n%s" % (pg["n"], pg.get("title") or pg["domain"], pg["url"], pg["excerpt"]))
    return "\n\n".join(lines)


def judge_block(run_id):
    """Same excerpts for the judge, so [X:n] citations in claims and verdicts can be checked."""
    d = os.path.join(RUNS, run_id)
    src = jload(os.path.join(d, "meta.json"), {}).get("web_run") or run_id
    nb = notes_block(src)
    return nb.replace("WEB RESEARCH (", "WEB SOURCES SEEN BY THE GENERATOR (", 1) if nb else ""


def plan_queries(server, task_prompt, seed=7):
    """Ask the generator (thinking off) for search queries; JSON-schema constrained."""
    tp = server.apply_template(PLAN_PROMPT.format(n=MAX_QUERIES, task=task_prompt), thinking=False)
    body = {"prompt": tp, "n_predict": 400, "temperature": 0.0, "seed": seed, "cache_prompt": False,
            "json_schema": PLAN_SCHEMA}
    from .harness import http_json
    res = http_json(server.base + "/completion", body, timeout=900)
    try:
        q = json.loads(re.search(r"\{.*\}", res.get("content", ""), re.S).group(0)).get("queries", [])
    except Exception:
        q = []
    return [s.strip() for s in q if isinstance(s, str) and s.strip()][:MAX_QUERIES]
