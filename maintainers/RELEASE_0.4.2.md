# Shipping ASHA 0.4.2 (developer-preview production-pilot build)

Stay on version **0.4.2** (do not bump for this phase).

Default install for everyone (lite):

```bash
pip install asha-si
```

Optional pin for long-lived pilots that need a frozen dependency:

```bash
pip install asha-si==0.4.2
```

## Product stance (who this build is for)

| Audience | Guidance |
|----------|----------|
| Everyone | `pip install asha-si` (lite path, no API key, no models) |
| LLM app authors | `wrap_llm` / `process`; pin only when you need a frozen pilot |
| Agent authors | `anchor(..., interactive=False)`: validate on your stack (preview) |
| Hardened operators | `asha-si[hardened]`: models auto-download; see `maintainers/PRIVATE_ARTIFACTS.md` |

Do **not** market 0.4.2 as semver-stable 1.0. Do market it as: *the current developer-preview release; pin when you need reproducibility*.

## Pre-flight (must be green)

1. [ ] Rotate any exposed provider keys: `maintainers/SECRET_ROTATION.md`
2. [ ] `python scripts/check_public_surface.py`
3. [ ] `pytest tests/architecture tests/imports tests/public_api -q`
4. [ ] `pytest -m "not slow and not integration" -q` (or full CI)
5. [ ] Quickstart from README works in a clean venv:
   ```bash
   pip install asha-si
   python -c "from asha import process; print(process('Contact a@b.com').output)"
   ```
6. [ ] Public claims audit: README, `docs/status.md`, `docs/index.md`, PyPI description say developer preview; pin `asha-si==0.4.2`
7. [x] Community links: Discussions enabled; Slack invite published
8. [ ] No `training/`, `models/`, `data/`, `.env` in the commit

## Trusted publishing (required before Actions can upload)

GitHub environment **`pypi`** already exists. After the repo rename to `ASHA`, PyPI must trust this exact publisher (pending publisher creates project `asha-si` on first upload):

1. Open https://pypi.org/manage/account/publishing/
2. Under **GitHub**, add a pending publisher:
   - **PyPI project name:** `asha-si`
   - **Owner:** `AjayRajan05`
   - **Repository:** `ASHA`
   - **Workflow file name:** `publish.yml`
   - **Environment name:** `pypi`
3. Save, then re-run the failed **Publish to PyPI** workflow for `v0.4.2` (or recreate the GitHub Release).

Manual fallback (API token, never commit it):

```powershell
$env:TWINE_USERNAME = '__token__'
$env:TWINE_PASSWORD = 'pypi-...'   # create at https://pypi.org/manage/account/token/
python -m twine upload dist/asha_si-0.4.2*
```

## Publish sequence (when you choose to cut the GitHub/PyPI release)

```bash
python scripts/check_public_surface.py
pip install -e ".[dev]" build twine
python -m build
twine check dist/asha_si-*
# tag only after CI green on main
git tag -a v0.4.2 -m "ASHA 0.4.2 developer preview (production-pilot pin)"
git push origin v0.4.2
# GitHub Release / trusted publishing as configured in .github
```

If `v0.4.2` already exists on the remote, do **not** retag unless the publish commit changed: keep the pin `asha-si==0.4.2` (no version bump per project decision).

## After publish

- [ ] Docs site (`mkdocs gh-deploy` or Actions) matches `docs/` on the release commit
- [ ] PyPI page shows developer-preview description from `pyproject.toml`
- [ ] Announcement: pin `asha-si==0.4.2`; link `docs/status.md` for stable vs experimental
