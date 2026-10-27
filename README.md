# roc-fusion-harness

Grounded, detector-fused hallucination prevention for a local open-weight LLM on Apple Silicon. This is the frozen harness from Program 2 of the roc-book research line, version 1.0.1.

## BLUF

On 24 unseen evaluation passages (12 Greek New Testament, 12 Hebrew Bible), the harness cut Qwen3.5-122B's hallucination rate in original-language exegesis from 12.7% in first drafts to **2.8% delivered** (95% CI 1.8–3.8%; target < 30%), while keeping **92.5%** of the true claims. Everything runs on one Mac Studio (M5 Ultra, 96 GB); only claim extraction and judging for evaluation use a frontier model.

| Split | Draft HR | Delivered HR (95% CI) | One-sided 95% upper bound | True claims kept |
| --- | --- | --- | --- | --- |
| All 24 passages | 12.7% | 2.8% (1.8–3.8) | 3.7% | 92.5% |
| Greek 12 | 15.2% | 3.3% (1.8–4.5) | 4.4% | 90.9% |
| Hebrew 12 | 10.2% | 2.4% (0.9–3.9) | 3.7% | 93.9% |

HR = hallucination rate, the share of a delivered answer's checkable claims that a judge rates false. Per-passage results are in `scripts/stageD/results/`.

## How it works

1. **Grounding (G3).** The generator gets a source packet with the original text, critical apparatus, morphology, lexicon entries, occurrence counts, Septuagint (LXX) parallels and witness tables, all from open data.
2. **Generation.** Qwen3.5-122B-A10B (Q4_K_S GGUF) on llama.cpp 0.7.0 build 11669, reasoning capped at 8,000 tokens, answers capped at 16,384. Web research is off in the frozen configuration.
3. **Detectors.**
   - V1: a deterministic source and morphology checker; contradictions with the packet are always flagged.
   - V3: token-level confidence from the generator's log-probabilities.
   - V5: a cross-family judge (Gemma 4 26B) that estimates each claim's chance of being false.
4. **Fusion.** A claim goes to repair only when **both** V3 and V5 flag it (Boolean AND; thresholds V3 0.399, V5 0).
5. **Repair and cut.** One repair pass against the same G3 packet, a re-check, and anything still flagged is cut.

Every setting that can change an answer — model weights (SHA-256), engine build, prompts, thresholds, and code hashes — is pinned in `harness.lock`.

## Repository layout

| Path | Contents |
| --- | --- |
| `p2/` | The harness: texts and grounding packets, generation, extraction, detectors, fusion, repair, judging |
| `harness.lock` | The frozen configuration (v1.0.1) |
| `prereg/` | Pre-registered passage frame and train/calibration/evaluation splits |
| `scripts/stageD/` | The verification run's per-passage results and report script |
| `tests/` | Unit and smoke tests |

## Requirements

- macOS on Apple Silicon with enough unified memory for a ~74 GB model (tested on a 96 GB M5 Ultra).
- `llama-server` from llama.cpp 0.7.0, build 11669 (Homebrew). Output hashes change across builds, so pin it.
- Model weights: Qwen3.5-122B-A10B Q4_K_S GGUF (generator) and Gemma 4 26B (V5 judge); hashes are in `harness.lock`.
- Python 3.12 with `numpy`, `matplotlib` and `pytest` (`pip install -r requirements.txt`).
- Open data, downloaded separately: SBLGNT with apparatus (CC BY 4.0), MorphGNT (CC BY-SA), OSHB/WLC (CC BY 4.0), Robinson–Pierpont (public domain), Swete LXX from First1KGreek (CC BY-SA 4.0), OpenBible.info cross-references (CC BY). No licensed commercial text is used or included.
- For evaluation only: claim extraction and judging ran on Claude Opus 5.5 through the Claude Code CLI.

Data, runs and state live outside the repo. Set `P2_ROOT` to that directory (default `~/llm_tests/program2`).

## Status and caveats

This is research code released as frozen, not a maintained package. The results come from one model, one judge and one domain (Biblical exegesis in Greek and Hebrew); expect to re-tune detector thresholds for another model or domain. The 27B Qwen3.8 model needs multi-token prediction off, because MTP zeroes the V3 signal.

## License

Apache License 2.0 — see `LICENSE` and `NOTICE`. Copyright 2026 Steven N. Thorsen.
