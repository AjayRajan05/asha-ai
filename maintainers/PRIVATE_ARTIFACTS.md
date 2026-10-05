# Private / auto-downloaded detector artifacts

## Decision (Phase 2.1: recommended option A)

| Path | Who | What |
|------|-----|------|
| **`pip install asha-ai`** | Everyone (default) | Lite/heuristic security. **No download required.** |
| **`pip install asha-ai[hardened]`** | Advanced users | Classical ML detectors. On first use, ASHA **automatically downloads** the official `asha-models-0.4.2` bundle into the user cache (**no permission prompt**). |
| **`ASHA_MODELS_DIR`** | Optional override | Point at a custom / air-gapped models tree instead of the cache. |

Public git tree still must **not** commit raw `training/`, datasets, or unpackaged `models/` working copies. Keep those **locally** (they are listed in `.gitignore`); the shippable bundle is the GitHub Release asset.

---

## How auto-download works

1. Default URL:  
   `https://github.com/AjayRajan05/ASHA/releases/download/asha-models-0.4.2/asha-models-0.4.2.zip`
2. Extracted to: `~/.cache/asha/models` (or `$XDG_CACHE_HOME/asha/models`, or `ASHA_MODELS_CACHE`)
3. Marker file: `.asha-models-0.4.2.ok`
4. No interactive consent: this is library-owned fetch for the hardened path.
5. If download fails → clear warning + **lite/heuristic fallback** (never silent “secure”).

### Env knobs

| Variable | Purpose |
|----------|---------|
| `ASHA_MODELS_DIR` | Use this directory; skip cache layout |
| `ASHA_MODELS_URL` | Override download URL (`file:` URLs supported) |
| `ASHA_MODELS_ZIP` | Local zip path fallback |
| `ASHA_MODELS_CACHE` | Override cache directory |
| `ASHA_DISABLE_MODEL_DOWNLOAD=1` | CI / air-gap: never fetch |

---

## Maintainer: build & publish the bundle

**Prefer real training** (see [TRAINING.md](TRAINING.md)), not the bootstrap stub:

```bash
python -m pip install -e ".[hardened]"
python training/train_hardened.py --pack --install-cache
# writes models/ from asha_training_data_generators, packs dist/asha-models-0.4.2.zip

gh release create asha-models-0.4.2 dist/asha-models-0.4.2.zip \
  --title "ASHA models 0.4.2" \
  --notes "Detector artifacts for asha 0.4.2 (auto-downloaded by model_store)."
```

`scripts/bootstrap_models.py` is **plumbing only** (`source: bootstrap_synthetic`). Do not publish it as the official hardened pack once `train_hardened.py` has been run.

Rebuild after detector retrains; keep the tag/version aligned with the library line (`0.4.2`).

---

## Layout inside the bundle

```text
injection/
  fusion_meta.joblib
  embedding_rf.joblib
  perplexity_gb.joblib
  lite/hashed_clf.json
  lite/char_ngram_lm.json
memory_guard/
  poisoning_rf.joblib
  lite/hashed_clf.json
chain_guard/
  transition_matrix.json
mission/
  domain_clf.joblib
intent/
  intent_clf.joblib
alignment/
  isotonic.joblib
  isotonic_breakpoints.json
```

---

## Public messaging

- Everyone: `pip install asha-ai` (works offline, no models).
- Hardened: `pip install asha-ai[hardened]` (models download automatically on first hardened detect).
- Never document exact threshold cutoffs or training recipes in public `docs/`.
