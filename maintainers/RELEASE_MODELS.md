# Release models bundle (maintainer)

Upload `dist/asha-models-0.4.2.zip` so `asha-si[hardened]` auto-download works.

## One-time: log in

```powershell
gh auth login
# GitHub.com → HTTPS → Login with a web browser → paste code
gh auth status
```

## Create / update the models release

From the repo root (`S:\Projects\privySHA\project`):

```powershell
# If the release does not exist yet:
gh release create asha-models-0.4.2 dist/asha-models-0.4.2.zip `
  --title "ASHA models 0.4.2" `
  --notes "Detector artifacts for asha 0.4.2 (auto-downloaded by model_store)."

# If the release already exists, only refresh the asset:
gh release upload asha-models-0.4.2 dist/asha-models-0.4.2.zip --clobber
```

Verify:

```powershell
gh release view asha-models-0.4.2
# Browser: https://github.com/AjayRajan05/asha-si/releases/download/asha-models-0.4.2/asha-models-0.4.2.zip
```

Default download URL (must return HTTP 200):

`https://github.com/AjayRajan05/asha-si/releases/download/asha-models-0.4.2/asha-models-0.4.2.zip`
