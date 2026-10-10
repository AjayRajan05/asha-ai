# Maintainers

Internal checklists for the **public** ASHA repo. End users should read [`docs/`](../docs/index.md), not this folder.

| Doc | Purpose |
|-----|---------|
| [SECRET_ROTATION.md](SECRET_ROTATION.md) | Rotate exposed provider keys |
| [PRIVATE_ARTIFACTS.md](PRIVATE_ARTIFACTS.md) | Lite-first + **auto-download** `asha-models` bundle |
| [TRAINING.md](TRAINING.md) | Train real `asha-si[hardened]` weights from generators (not bootstrap) |
| [RELEASE_0.4.2.md](RELEASE_0.4.2.md) | Ship / pin `0.4.2` without a version bump |

Public gate:

```bash
python scripts/check_public_surface.py
```
