# Support

ASHA is an open-source **developer preview** (`0.4.x`). This document is the support policy for the public project.

## Where to get help

| Need | Channel |
|------|---------|
| Usage questions, design discussion | [GitHub Discussions](https://github.com/AjayRajan05/ASHA/discussions) |
| Bug reports / feature requests | [GitHub Issues](https://github.com/AjayRajan05/ASHA/issues) (use the templates) |
| Chat (optional) | [ASHA Slack](https://join.slack.com/t/asha-community/shared_invite/zt-45h11ellh-yY~YG4BL8W~TvxsMXSgPzw) |
| Security vulnerability | [SECURITY.md](SECURITY.md): **email only**, not public issues |
| Contributing / PRs | [CONTRIBUTING.md](CONTRIBUTING.md) · [docs/contributing.md](docs/contributing.md) |
| Docs | [docs/index.md](docs/index.md) · [docs/faq.md](docs/faq.md) · [docs/troubleshooting.md](docs/troubleshooting.md) · [published site](https://ajayrajan05.github.io/ASHA/) |

## Support policy (preview)

- **Best-effort, community-driven.** Maintainers triage Issues and Discussions when available; there is **no SLA** and **no paid support contract** for the open-source package.
- **Pin versions** for pilots (`asha==0.4.2`). Preview APIs may change within `0.4.x` with CHANGELOG notes.
- **Default install** is lite (`pip install asha-ai`). Hardened models and framework adapters are optional extras (see [docs/status.md](docs/status.md) and [docs/anchor.md](docs/anchor.md)).
- **Security reports** follow [SECURITY.md](SECURITY.md) (acknowledge within 48 hours; status update within 7 days: target, not a guarantee).
- **Out of scope:** debugging third-party LLM providers, private forks without a minimal repro, or requesting managed/cloud hosting (not offered).

## What to include in a report

1. ASHA version (`python -c "import asha; print(asha.__version__)"`)
2. Python version and OS
3. Install extras used (`asha-ai[hardened]`, `asha-ai[anchor]`, …)
4. Minimal reproduction (code + expected vs actual)
5. Whether `ASHA_DISABLE_ML` / `ASHA_DISABLE_MODEL_DOWNLOAD` / `ASHA_MODELS_DIR` are set

## Community links

- [GitHub Discussions](https://github.com/AjayRajan05/ASHA/discussions)
- [ASHA Slack](https://join.slack.com/t/asha-community/shared_invite/zt-45h11ellh-yY~YG4BL8W~TvxsMXSgPzw)
- [GitHub Issues](https://github.com/AjayRajan05/ASHA/issues)

## Related

- Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- Project status: [docs/status.md](docs/status.md)
- Roadmap: [docs/roadmap.md](docs/roadmap.md)
