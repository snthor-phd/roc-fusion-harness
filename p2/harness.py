"""WP2: generation harness. Stacks S1 (Ollama + MLX, side-by-side 0.40.2) and S2 (llama-server + GGUF).

Reasoning factor (D20): 'off' = thinking disabled in the chat template; 'capped' = thinking generated with an
8,000-token limit; if no closing tag by then, a budget-forcing sentence and the closing tag are appended and the
answer is generated as a continuation. Uncapped reasoning is never run.
"""
import hashlib
import json
import math
import os
import signal
import subprocess
import time
import urllib.request

from .common import HOME, P2, STATE, LOGS, jdump, jload

LT = os.path.join(HOME, "llm_tests")
BIN = os.path.join(LT, "ollama-0.40.2", "bin")
# Stage A (19 Oct 2026): Homebrew llama.cpp 0.7.0 build 11669 (Metal tensor API on). Pinned for the frozen harness.
LLAMA_SERVER = os.environ.get("P2_LLAMA_SERVER", "/opt/homebrew/bin/llama-server")
OLLAMA = os.path.join(BIN, "ollama")
OLLAMA_HOST = "127.0.0.1:11435"
OLLAMA_MODELS = os.path.join(LT, "ollama-0.40.2", "models")
PINNED_OLLAMA = "http://127.0.0.1:11434"
LLAMA_PORT = 8130
# v1.0.1: when set (e.g. http://127.0.0.1:8100), the harness is a client of ultra-llm-svc's gateway. It never
# starts or kills a llama-server; it uses the resident server and refuses any generator whose weights it isn't.
GATEWAY = os.environ.get("P2_LLAMA_BASE", "").rstrip("/")
GATEWAY_HEADERS = {"X-Workload": "exegesis", "X-Prompt-Version": "harness-v1.0.1"}
# v1.0.1 lease mode (P2_LEASE=1): a generator the gateway doesn't serve (V5's Gemma) borrows the machine under a
# gateway lease, runs exactly as v1.0.0 does (its own llama-server on LLAMA_PORT), then releases it.
GATEWAY_GEN = os.environ.get("P2_GATEWAY_GEN", "qwen35_122b")  # the generator the gateway keeps resident
LEASE = os.environ.get("P2_LEASE") == "1"
LEASE_FORCE = os.environ.get("P2_LEASE_FORCE") == "1"         # outside the 01:00-05:00 window; Thor only
LEASE_MAX_S = int(os.environ.get("P2_LEASE_MAX_S", "10800"))
LEASE_HOLDER = os.environ.get("P2_LEASE_HOLDER", "program2-v5")

THINK_CAP = int(os.environ.get("P2_THINK_CAP", "8000"))
MIN_THINK = int(os.environ.get("P2_MIN_THINK", "0"))  # budget forcing: 0 = off (v1.0.0)
MAX_FORCES = int(os.environ.get("P2_MAX_FORCES", "6"))
WAIT = "\n\nWait,"
ANSWER_MAX = int(os.environ.get("P2_ANSWER_MAX", "16384"))  # 12,288 in Phase 0 (J21); 16,384 from Stage B
FORCE_CLOSE = "\n\nConsidering the limited time, I have to give the final answer based on the thinking so far.\n"

QWEN_SAMPLING = {"capped": dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0),
                 "off": dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0)}
GEMMA_SAMPLING = {"capped": dict(temperature=1.0, top_p=0.95, top_k=64, min_p=0.0),
                  "off": dict(temperature=1.0, top_p=0.95, top_k=64, min_p=0.0)}

GENERATORS = {
    "qwen38_27b": {"label": "Qwen3.8-27B UD-Q4_K_M", "stack": "S2", "family": "qwen",
                   "gguf": os.path.join(LT, "q38gguf", "Qwen3.8-27B-UD-Q4_K_M.gguf"),
                   "args": ["--n-gpu-layers", "999", "-c", "40960", "--spec-type", "draft-mtp"],
                   "sampling": QWEN_SAMPLING},
    "qwen36_35b": {"label": "Qwen3.6-35B-A3B UD-Q4_K_M", "stack": "S2", "family": "qwen",
                   "gguf": os.path.join(P2, "models", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
                   "args": ["--n-gpu-layers", "999", "-c", "32768"], "sampling": QWEN_SAMPLING},
    "gemma4_26b": {"label": "Gemma 4 26B-A4B it UD-Q4_K_M", "stack": "S2", "family": "gemma",
                   "gguf": os.path.join(P2, "models", "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"),
                   "args": ["--n-gpu-layers", "999", "-c", "32768"], "sampling": GEMMA_SAMPLING},
    "qwen35_122b": {"label": "Qwen3.5-122B-A10B Q4_K_S", "stack": "S2", "family": "qwen",
                    "gguf": os.path.join(LT, "q122b", "Qwen3.5-122B-A10B-Q4_K_S-00001-of-00003.gguf"),
                    "args": ["--n-gpu-layers", "999", "-c", "40960"], "sampling": QWEN_SAMPLING},
    "qwen36_mlx": {"label": "qwen3.6:35b-mlx (Ollama 0.40.2 MLX)", "stack": "S1", "family": "qwen",
                   "ollama_model": "qwen3.6:35b-mlx", "sampling": QWEN_SAMPLING},
}

# thinking delimiters per family; gemma filled in from config/think_tags.json after the template probe
THINK_TAGS = {"qwen": ("<think>", "</think>"), "gemma": None}
_tt = jload(os.path.join(STATE, "think_tags.json"), {})
for _k, _v in _tt.items():
    THINK_TAGS[_k] = tuple(_v) if _v else None


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    with open(os.path.join(LOGS, "harness.log"), "a") as f:
        f.write(line + "\n")


def http_json(url, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if GATEWAY and url.startswith(GATEWAY):
        headers.update(GATEWAY_HEADERS)
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def file_hash(path):
    """sha256 (first 16 hex) of the weight file(s), cached by (path, size, mtime)."""
    cache_p = os.path.join(STATE, "weight_hashes.json")
    cache = jload(cache_p, {})
    paths = [path]
    if "-00001-of-" in path:
        n = int(path.split("-of-")[1].split(".")[0])
        paths = [path.replace("-00001-of-", "-%05d-of-" % i) for i in range(1, n + 1)]
    out = []
    for p in paths:
        st = os.stat(p)
        key = "%s|%d|%d" % (p, st.st_size, int(st.st_mtime))
        if key not in cache:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 24), b""):
                    h.update(chunk)
            cache[key] = h.hexdigest()
            jdump(cache, cache_p)
        out.append(cache[key][:16])
    return "+".join(out)


def llama_version():
    try:
        r = subprocess.run([LLAMA_SERVER, "--version"], capture_output=True, text=True, timeout=30)
        lines = [l for l in (r.stdout + r.stderr).splitlines() if "version" in l.lower()]
        return (lines[-1] if lines else (r.stdout + r.stderr).strip()[-120:])[:120]
    except Exception as e:
        return "unknown (%s)" % e


# ------------------------------------------------------------------ llama-server (S2)
class LlamaServer:
    def __init__(self, gen_key, extra_args=None):
        self.key = gen_key
        self.cfg = GENERATORS[gen_key]
        self.proc = None
        self.extra = extra_args or []
        self.lease_id = None
        self.base = GATEWAY or "http://127.0.0.1:%d" % LLAMA_PORT

    def _attach(self, timeout):
        """Gateway mode: wait for the resident server, then check it is this generator's model."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if http_json(self.base + "/health", timeout=10).get("status") == "ok":
                    break
            except Exception:
                pass
            time.sleep(10)
        else:
            raise RuntimeError("gateway not healthy after %ds" % timeout)
        mp = (self.props().get("model_path") or "")
        if os.path.basename(mp) != os.path.basename(self.cfg["gguf"]):
            raise RuntimeError("gateway serves %s, generator %s needs %s" % (mp, self.key, self.cfg["gguf"]))
        return time.time() - t0

    def start(self, timeout=1800):
        if GATEWAY:
            if self.key == GATEWAY_GEN:
                return self._attach(timeout)
            if not LEASE:
                raise RuntimeError("generator %s isn't served by the gateway; V5 needs P2_LEASE=1" % self.key)
            self._lease_acquire()
            try:
                return self._spawn(timeout)
            except Exception:
                self._lease_release()
                raise
        kill_llama_servers()
        return self._spawn(timeout)

    def _lease_acquire(self):
        body = {"holder": LEASE_HOLDER, "max_s": LEASE_MAX_S, "force": LEASE_FORCE}
        r = http_json(GATEWAY + "/admin/lease", body, timeout=1800)
        self.lease_id = r["lease_id"]
        log("lease %s granted to %s until %s (drain %ss, unload %ss)" % (self.lease_id, LEASE_HOLDER,
            time.ctime(r["until"]), r.get("drain_s"), r.get("unload_s")))

    def _lease_release(self):
        if not self.lease_id:
            return
        lid, self.lease_id = self.lease_id, None
        for attempt in range(3):
            try:
                r = http_json(GATEWAY + "/admin/lease/%s/release" % lid, {}, timeout=1200)
                log("lease %s released; 122B back after %ss" % (lid, r.get("reload_s")))
                return
            except Exception as e:
                log("lease %s release failed (%s); retrying" % (lid, e))
                time.sleep(15)
        log("lease %s NOT released; the gateway keeps holding and will alert" % lid)

    def _spawn(self, timeout):
        self.base = "http://127.0.0.1:%d" % LLAMA_PORT
        cmd = [LLAMA_SERVER, "-m", self.cfg["gguf"], "--host", "127.0.0.1", "--port", str(LLAMA_PORT),
               "--jinja", "--reasoning-format", "none", "-np", "1"] + self.cfg["args"] + self.extra
        logf = open(os.path.join(LOGS, "llama_server_%s.log" % self.key), "a")
        logf.write("\n==== %s %s\n" % (time.ctime(), " ".join(cmd)))
        logf.flush()
        self.proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server exited with %s" % self.proc.returncode)
            try:
                if http_json(self.base + "/health", timeout=5).get("status") == "ok":
                    return time.time() - t0
            except Exception:
                pass
            time.sleep(3)
        raise RuntimeError("llama-server start timeout")

    def stop(self):
        if GATEWAY and not self.lease_id:
            return  # the gateway owns the model
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except Exception:
                self.proc.terminate()
            try:
                self.proc.wait(60)
            except Exception:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except Exception:
                    pass
        self.proc = None
        self._lease_release()

    def props(self):
        return http_json(self.base + "/props", timeout=30)

    def apply_template(self, prompt, thinking):
        body = {"messages": [{"role": "user", "content": prompt}],
                "chat_template_kwargs": {"enable_thinking": bool(thinking)}}
        return http_json(self.base + "/apply-template", body, timeout=120)["prompt"]

    def tokenize_count(self, text):
        return len(http_json(self.base + "/tokenize", {"content": text}, timeout=120).get("tokens", []))

    def complete(self, prompt, n_predict, sampling, seed, stop=None, n_probs=0, timeout=6 * 3600):
        body = {"prompt": prompt, "n_predict": n_predict, "seed": seed, "cache_prompt": True,
                "stream": False, "n_probs": n_probs}
        body.update(sampling)
        if stop:
            body["stop"] = stop
        return http_json(self.base + "/completion", body, timeout=timeout)


def kill_llama_servers():
    if GATEWAY:
        return  # never kill the service's resident model
    subprocess.run(["pkill", "-f", LLAMA_SERVER], capture_output=True)
    time.sleep(2)


def _token_logprobs(resp):
    """Normalize llama.cpp completion_probabilities into [(token, logprob)]."""
    out = []
    for item in resp.get("completion_probabilities") or []:
        tok = item.get("token", item.get("content", ""))
        lp = None
        if "logprob" in item:
            lp = item["logprob"]
        elif "prob" in item:
            lp = math.log(max(item["prob"], 1e-12))
        else:
            for pr in item.get("probs") or item.get("top_logprobs") or []:
                if pr.get("tok_str", pr.get("token")) == tok:
                    lp = pr.get("logprob")
                    if lp is None and "prob" in pr:
                        lp = math.log(max(pr["prob"], 1e-12))
        out.append((tok, lp))
    return out


def _stopped_by_word(r):
    return r.get("stop_type") == "word" or bool(r.get("stopped_word"))


def generate_llama(server, gen_key, prompt, reasoning, seed):
    cfg = GENERATORS[gen_key]
    fam = cfg["family"]
    samp = cfg["sampling"][reasoning]
    meta = {"reasoning": reasoning, "seed": seed, "sampling": samp}
    if GATEWAY and getattr(server, "base", "") == GATEWAY:
        meta["via"] = "ultra-llm-svc gateway " + GATEWAY
    tags = THINK_TAGS.get(fam)
    t0 = time.time()
    thinking = ""
    if reasoning == "off":
        tp = server.apply_template(prompt, thinking=False)
        meta["template_tail"] = tp[-80:]
        r = server.complete(tp, ANSWER_MAX, samp, seed, n_probs=1)
        answer = r.get("content", "")
        lps = _token_logprobs(r)
        meta.update(answer_tokens=r.get("tokens_predicted"), prompt_tokens=r.get("tokens_evaluated"),
                    answer_stop=r.get("stop_type"), think_tokens=0,
                    gen_tok_per_s=(r.get("timings") or {}).get("predicted_per_second"),
                    prompt_tok_per_s=(r.get("timings") or {}).get("prompt_per_second"))
        if tags and tags[1] in answer:
            pre, post = answer.split(tags[1], 1)
            thinking, answer = pre.replace(tags[0], ""), post.lstrip()
            meta["note"] = "model emitted a think block despite thinking off"
            lps = []  # cannot align; V3 unavailable for this run
    else:
        if not tags:
            raise RuntimeError("no thinking delimiters configured for family %s" % fam)
        tp = server.apply_template(prompt, thinking=True)
        meta["template_tail"] = tp[-80:]
        r1 = server.complete(tp, THINK_CAP, samp, seed, stop=[tags[1]])
        thinking = r1.get("content", "")
        n_think = r1.get("tokens_predicted", 0)
        closed = _stopped_by_word(r1)
        forces = 0
        while MIN_THINK and closed and n_think < MIN_THINK and forces < MAX_FORCES and n_think < THINK_CAP:
            pre = "" if (tp.rstrip().endswith(tags[0]) or tags[0] in thinking) else tags[0] + "\n"
            thinking = thinking + WAIT
            rx = server.complete(tp + pre + thinking, THINK_CAP - n_think, samp, seed + 2 + forces, stop=[tags[1]])
            thinking += rx.get("content", "")
            n_think += rx.get("tokens_predicted", 0)
            closed = _stopped_by_word(rx)
            forces += 1
        meta["budget_forces"] = forces
        prefix = "" if (tp.rstrip().endswith(tags[0]) or tags[0] in thinking) else tags[0] + "\n"
        if closed:
            cont = tp + prefix + thinking + tags[1] + "\n\n"
        else:
            cont = tp + prefix + thinking + FORCE_CLOSE + tags[1] + "\n\n"
        r2 = server.complete(cont, ANSWER_MAX, samp, seed + 1, n_probs=1)
        answer = r2.get("content", "")
        lps = _token_logprobs(r2)
        meta.update(think_tokens=n_think, think_closed_naturally=closed, answer_tokens=r2.get("tokens_predicted"),
                    prompt_tokens=r1.get("tokens_evaluated"), answer_stop=r2.get("stop_type"),
                    think_tok_per_s=(r1.get("timings") or {}).get("predicted_per_second"),
                    prompt_tok_per_s=(r1.get("timings") or {}).get("prompt_per_second"),
                    gen_tok_per_s=(r2.get("timings") or {}).get("predicted_per_second"))
        thinking = thinking.replace(tags[0], "")
    meta["wall_s"] = round(time.time() - t0, 1)
    return answer, thinking, lps, meta


# ------------------------------------------------------------------ Ollama MLX (S1)
_ollama_proc = None


def ollama_up():
    try:
        http_json("http://%s/api/version" % OLLAMA_HOST, timeout=5)
        return True
    except Exception:
        return False


def start_ollama():
    global _ollama_proc
    if ollama_up():
        return
    env = dict(os.environ, OLLAMA_HOST=OLLAMA_HOST, OLLAMA_MODELS=OLLAMA_MODELS)
    logf = open(os.path.join(LOGS, "ollama_0402.log"), "a")
    _ollama_proc = subprocess.Popen([OLLAMA, "serve"], env=env, stdout=logf, stderr=subprocess.STDOUT,
                                    start_new_session=True)
    for _ in range(60):
        if ollama_up():
            return
        time.sleep(2)
    raise RuntimeError("ollama 0.40.2 did not start")


def stop_ollama():
    global _ollama_proc
    if not ollama_up():
        return
    try:
        for m in http_json("http://%s/api/ps" % OLLAMA_HOST, timeout=10).get("models", []):
            http_json("http://%s/api/generate" % OLLAMA_HOST, {"model": m["name"], "keep_alive": 0}, timeout=60)
    except Exception:
        pass
    if _ollama_proc and _ollama_proc.poll() is None:
        try:
            os.killpg(_ollama_proc.pid, signal.SIGTERM)
            _ollama_proc.wait(60)
        except Exception:
            pass
    _ollama_proc = None


def qwen_chatml(prompt, thinking):
    s = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n" % prompt
    if thinking:
        return s + "<think>\n"
    return s + "<think>\n\n</think>\n\n"


def _ollama_gen(model, prompt, num_predict, samp, seed, stop=None, logprobs=False):
    opts = {"num_predict": num_predict, "seed": seed, "num_ctx": 32768,
            "temperature": samp["temperature"], "top_p": samp["top_p"], "top_k": samp["top_k"], "min_p": samp["min_p"]}
    if stop:
        opts["stop"] = stop
    body = {"model": model, "prompt": prompt, "raw": True, "stream": False, "options": opts, "keep_alive": "10m"}
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 1
    return http_json("http://%s/api/generate" % OLLAMA_HOST, body, timeout=6 * 3600)


def generate_ollama(gen_key, prompt, reasoning, seed):
    cfg = GENERATORS[gen_key]
    model = cfg["ollama_model"]
    samp = cfg["sampling"][reasoning]
    meta = {"reasoning": reasoning, "seed": seed, "sampling": samp, "raw_template": "qwen chatml (harness)"}
    t0 = time.time()
    thinking = ""
    if reasoning == "off":
        r = _ollama_gen(model, qwen_chatml(prompt, False), ANSWER_MAX, samp, seed, logprobs=True)
        answer = r.get("response", "")
        meta.update(answer_tokens=r.get("eval_count"), prompt_tokens=r.get("prompt_eval_count"), think_tokens=0)
    else:
        tp = qwen_chatml(prompt, True)
        r1 = _ollama_gen(model, tp, THINK_CAP, samp, seed, stop=["</think>"])
        thinking = r1.get("response", "")
        n_think = r1.get("eval_count", 0) or 0
        closed = r1.get("done_reason") == "stop" and n_think < THINK_CAP
        cont = tp + thinking + ("" if closed else FORCE_CLOSE) + "</think>\n\n"
        r = _ollama_gen(model, cont, ANSWER_MAX, samp, seed + 1, logprobs=True)
        answer = r.get("response", "")
        meta.update(think_tokens=n_think, think_closed_naturally=closed, answer_tokens=r.get("eval_count"),
                    prompt_tokens=r1.get("prompt_eval_count"),
                    think_tok_per_s=(n_think / (r1["eval_duration"] / 1e9)) if r1.get("eval_duration") else None)
    ed = r.get("eval_duration")
    meta["gen_tok_per_s"] = (r.get("eval_count", 0) / (ed / 1e9)) if ed else None
    meta["answer_stop"] = r.get("done_reason")
    lps = [(x.get("token", ""), x.get("logprob")) for x in (r.get("logprobs") or [])]
    meta["wall_s"] = round(time.time() - t0, 1)
    try:
        meta["ollama_digest"] = [m for m in http_json("http://%s/api/tags" % OLLAMA_HOST).get("models", [])
                                 if m["name"] == model][0]["digest"][:12]
        meta["engine"] = "ollama " + http_json("http://%s/api/version" % OLLAMA_HOST)["version"]
    except Exception:
        pass
    try:
        http_json("http://%s/api/generate" % OLLAMA_HOST, {"model": model, "keep_alive": 0}, timeout=120)
    except Exception:
        pass
    return answer, thinking, lps, meta


# ------------------------------------------------------------------ priority guard
IDLE_MIN = float(os.environ.get("P2_IDLE_MIN", "5"))  # minutes of no keyboard/mouse before starting work
NIGHT = (23, 7)  # 11 PM to 7 AM: no idle requirement


def hid_idle_s():
    try:
        out = subprocess.run(["ioreg", "-c", "IOHIDSystem"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if "HIDIdleTime" in line:
                return int(line.split("=")[-1].strip()) / 1e9
    except Exception:
        pass
    return None


def thor_busy():
    """Reasons to yield: Thor is at the keyboard (daytime), pinned Ollama has a model loaded (Open WebUI use),
    the transcriber is busy, or we are inside one of Thor's scheduled-job windows."""
    reasons = []
    hr = time.localtime().tm_hour
    if IDLE_MIN > 0 and not (hr >= NIGHT[0] or hr < NIGHT[1]):
        idle = hid_idle_s()
        if idle is not None and idle < IDLE_MIN * 60:
            reasons.append("Thor active (idle %.0fs < %.0f min)" % (idle, IDLE_MIN))
    try:
        ps = http_json(PINNED_OLLAMA + "/api/ps", timeout=5).get("models", [])
        if ps:
            reasons.append("pinned ollama has %d model(s) loaded" % len(ps))
    except Exception:
        pass
    lt = time.localtime()
    hm = lt.tm_hour * 60 + lt.tm_min
    windows = [(6 * 60 + 25, 6 * 60 + 50), (9 * 60 + 35, 10 * 60 + 0)]
    if lt.tm_wday < 5:
        windows.append((16 * 60 + 25, 16 * 60 + 50))
    if lt.tm_mday == 1:
        windows.append((9 * 60 + 10, 9 * 60 + 35))
    for a, b in windows:
        if a <= hm < b:
            reasons.append("scheduled-job window %02d:%02d" % (a // 60, a % 60))
    try:
        out = subprocess.run(["ps", "-Ao", "pcpu,command"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            low = line.lower()
            if ("whisper" in low or "transcrib" in low) and "grep" not in low:
                try:
                    if float(line.split()[0]) > 50:
                        reasons.append("transcriber busy")
                except ValueError:
                    pass
    except Exception:
        pass
    return reasons
