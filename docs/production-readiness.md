# Production readiness (0.4.2)

ASHA remains **v0.4.2 developer preview**. This page supports **pinned pilots today** and records what a **future** 1.0 would need; it is **not** a plan to ship 1.0 soon.

Finish release leftovers and in-version work first. Do not bump the version for 1.0 until the maintainer is ready to stand behind a stable guarantee.

## Release setup status (0.4.2)

| Item | Status |
|------|--------|
| `asha-models-0.4.2` GitHub Release asset | Done: auto-download URL in [PRIVATE_ARTIFACTS](https://github.com/AjayRajan05/asha-si/blob/main/maintainers/PRIVATE_ARTIFACTS.md) |
| GitHub Discussions | Enabled: [discussions](https://github.com/AjayRajan05/asha-si/discussions) |
| Optional Slack invite | Live: [join](https://join.slack.com/t/asha-community/shared_invite/zt-45h11ellh-yY~YG4BL8W~TvxsMXSgPzw); see [support.md](support.md) |

Hardened installs download the official bundle on first use; override with `ASHA_MODELS_DIR` / `ASHA_MODELS_ZIP` / `ASHA_DISABLE_MODEL_DOWNLOAD=1` (see [status.md](status.md)).

## Production-pilot checklist (0.4.2 today)

- [ ] Pin `asha==0.4.2`
- [ ] Use lite path unless you validated hardened models locally or via cache
- [ ] ANCHOR: `interactive=False` in CI/scripts; validate adapters on your stack ([anchor.md](anchor.md) matrix)
- [ ] Read [security.md](security.md) fail-closed / opt-out flags
- [ ] No secrets or private `data/` / unreleased weights in deploy artifacts
- [ ] Support expectations: [support.md](support.md) (best-effort, no SLA)

## Future 1.0 (deferred)

Only consider **1.0.0** after 0.4.2 is complete and the maintainer is ready. All of the following would need to be true then:

1. **Models distribution**: default `asha-models` URL returns 200; clean `pip install asha-si[hardened]` works without local zip.
2. **API freeze window**: a deliberate deprecation-only period on the public root + `wrap_llm` surface.
3. **Community signal**: feedback channels used; no open P0 security issues without a documented mitigation.
4. **Docs freeze**: status, versioning, migration, and support docs match shipped behavior.
5. **CI green** on supported Python versions; architecture + ANCHOR golden-path jobs required.

### Public API candidates (eventual)

| Surface | Intent if/when 1.0 |
|---------|-------------------|
| `process` / `sanitize` / `optimize` | Stable |
| Typed results (`ProcessResult`, …) | Stable |
| `wrap_llm` | Stable (prefer over `auto_patch`) |
| `Agent` mock + basic providers | Stable within documented providers |
| `anchor()` / golden adapters | Preview until ready to graduate |
| Experimental (AshaFit, `auto_patch`, smart routing) | Stay experimental or graduate via CHANGELOG |

## Related

- [roadmap.md](roadmap.md)
- [versioning.md](versioning.md)
- [SUPPORT.md](https://github.com/AjayRajan05/asha-si/blob/main/SUPPORT.md)
- [GitHub Discussions](https://github.com/AjayRajan05/asha-si/discussions) · [Slack](https://join.slack.com/t/asha-community/shared_invite/zt-45h11ellh-yY~YG4BL8W~TvxsMXSgPzw)
