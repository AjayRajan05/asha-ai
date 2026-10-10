# Training hardened detection models (`asha-si[hardened]`)

This is the **real** training path for versioned classical ML/DL detectors.
It is **not** the same as `scripts/bootstrap_models.py`, which only writes synthetic
placeholder weights (`source: bootstrap_synthetic`) so the download/cache plumbing works.

**Local only:** `training/`, `asha_training_data_generators/`, `data/`, `models/`,
`jailbreak_llms/`, and helper scripts (`llm_augment.py`, `build_*.py`,
`improve_data_quality.py`) are **gitignored**. Keep them on your machine; they
must not appear in the public GitHub tree or sdist.

## What trains what

| Artifact | Script | Data (prefer generators) | Hardening § |
|----------|--------|--------------------------|-------------|
| Injection lite (hashed + char n-gram) | `training/injection/train_lite.py` | `injection/{malicious,benign}.jsonl` | 2.1 |
| Injection ensemble (ppl GB + MiniLM RF + fusion) | `training/injection/train.py` | same | 2.1 |
| Memory poisoning lite | `training/memory_guard/train_lite.py` | `memory_guard/{poisoning,benign}.jsonl` | 2.3 |
| Memory poisoning RF on MiniLM | `training/memory_guard/train.py` | + optional `sensitive.jsonl` | 2.3 |
| Chain Markov matrix | `training/chain_guard/train.py` | `chain_guard/transition_matrix.json` | 2.4 |
| Mission multi-label domains | `training/mission/train.py` | `mission/domain_examples.jsonl` | 2.5 |
| Intent (compile_prompt) | `training/intent/train.py` | templates + mission texts | 2.6 |
| Alignment isotonic | `training/alignment/train.py` | `alignment/labeled_actions.jsonl` | 2.7 |
| Thresholds YAML | `training/train_hardened.py` | reports from above | §3 |

Data resolution (`training/_paths.py`): `data/<…>` overlay wins, else
`asha_training_data_generators/training/data/<…>`.

PII Indian-format samples under `data/pii/` are for **regression tests / validators**,
not a separate trainable weight file in this pack.

## Prerequisites

```bash
# From repo root
python -m pip install -e ".[hardened]"
# Generators optional deps if you rebuild JSONL from templates:
python -m pip install faker
```

Offline note: first MiniLM encode downloads `sentence-transformers/all-MiniLM-L6-v2`
unless already cached by Hugging Face. No generative LLM APIs are used.

## Refresh synthetic corpus (optional)

```bash
cd asha_training_data_generators/training
python injection/build_dataset.py
python memory_guard/build_dataset.py
python mission/build_dataset.py
python pii/indian_formats/generate_synthetic.py
python alignment/build_dataset.py
python chain_guard/build_transition_model.py
```

Then **supplement** with public jailbreaks / scrubbed agent logs / red-team paraphrases
under repo-root `data/` (same layout). Do not replace the synthetic set entirely.

### Recommended quality loop (local `data/`, never commit)

```bash
pip install faker
git clone --depth 1 https://github.com/verazuo/jailbreak_llms.git
# Best local corpus: Dec-2023 train + May-2023 external eval (disjoint)
python build_best_dataset.py --jailbreak-repo ./jailbreak_llms --out ./data
python training/train_hardened.py --pack --install-cache
# inspect external held-out FPR/TPR:
#   models/injection/train_report.json
#   models/memory_guard/train_report.json
# publish weights only (not data/):
gh release upload asha-models-0.4.2 dist/asha-models-0.4.2.zip --clobber
```

Chain / mission / alignment are large synthetics until real logs/verdicts replace them.
Injection/memory report `meets_fpr_target` on an **external** held-out set.

## Train everything

```bash
# Full pack: train → write config/thresholds.yaml + src/asha/config/thresholds.yaml
# → dist/asha-models-0.4.2.zip → ~/.cache/asha/models
python training/train_hardened.py --pack --install-cache
```

Useful flags:

- `--skip-embedding`: skip MiniLM steps (injection/memory full trainers); still trains lite/chain/mission/intent/alignment
- `--pack`: write `dist/asha-models-0.4.2.zip`
- `--install-cache`: replace user cache so runtime loaders pick up new weights without download

Single-module:

```bash
python training/injection/train.py
python training/memory_guard/train.py
# …
```

## Bootstrap vs real train

| | `scripts/bootstrap_models.py` | `training/train_hardened.py` |
|--|-------------------------------|------------------------------|
| Purpose | CI / release plumbing stub | Production-ish detector weights |
| `train_report.json` `source` | `bootstrap_synthetic` | `asha_training_data_generators` |
| MiniLM / sklearn fit | No (toy) | Yes |
| Safe to ship as “hardened models” | No | Yes (after you add real red-team data + recalibrate) |

## After training

1. Inspect `models/*/train_report.json` (counts, derived thresholds, accuracy).
2. Publish `dist/asha-models-0.4.2.zip` as GitHub release asset `asha-models-0.4.2` (needs `gh auth login`).
3. Or point pilots at local cache via `ASHA_MODELS_DIR` / `ASHA_MODELS_ZIP`.

## Calibration caveat

Thresholds derived only on this template corpus will under-estimate real-world FPR.
Re-run `train_hardened.py` after merging held-out public + red-team labels before calling
the pack production-ready.
