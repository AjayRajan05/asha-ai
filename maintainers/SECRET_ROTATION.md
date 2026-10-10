# Secret rotation checklist (Phase 0)

Do this once per exposed credential. ASHA maintainers cannot revoke cloud keys from the repo; you must do it in the provider console.

## Known exposure (2026-07)

A local `.env` previously held Google Generative AI keys (`GEMINI_API_KEY` / `GOOGLE_API_KEY`). That file was deleted from the working tree and is gitignored. `.env` was **not** found in git history for this clone, but treat the keys as compromised if they were ever shared, pasted into chat, or synced to another machine.

### Google AI / Gemini

1. Open [Google AI Studio API keys](https://aistudio.google.com/app/apikey) (and Cloud Console credentials if used).
2. **Delete / revoke** the exposed key(s).
3. Create a new key; store only in local `.env` (from `.env.example`).
4. Update any CI secrets / deployment env vars to the new key.
5. Confirm old key returns unauthorized on a test call.

### Other providers

Rotate any key that ever lived in `.env`, chat logs, or screenshots: OpenAI, Anthropic, Grok, etc.

## Repo verification

```bash
python scripts/check_public_surface.py
```

Must pass before tagging or publishing `0.4.2`.

## Rules going forward

- Never commit `.env` (gitignored; `.env.example` is the only template).
- Prefer OS keychain / CI secrets for real keys.
- Core library users need **no** API key for `process()` / `sanitize()` / `optimize()` / mock `Agent` / headless `anchor()`.
