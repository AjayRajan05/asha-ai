# Security Policy

## Supported Versions

| Version | Supported | Notes |
| ------- | --------- | ----- |
| 0.4.x   | :white_check_mark: | Current developer preview (pin exact version) |
| 0.3.x   | :x: | Superseded |
| 1.0.x   | :x: | Retracted track (brief 2026-05 line; project returned to 0.x) |
| < 0.3   | :x: | Unsupported |

## Reporting a Vulnerability

If you discover a security vulnerability in ASHA, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, email: **ajayrajan727@gmail.com**

Include:

- A description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We aim to acknowledge reports within **48 hours** and provide a status update within **7 days**.

## Scope

This policy covers:

- PII detection and masking bypasses
- Prompt injection / jailbreak evasion in ASHA's security layer
- Fail-safe behavior failures (data leakage on error paths)
- Dependency vulnerabilities in core `asha` package dependencies

Out of scope:

- Vulnerabilities in third-party LLM providers (OpenAI, Anthropic, etc.)
- Applications that disable privacy features (`privacy=False`, `mode="off"`)
- ML model behavior when using optional `[ml]` / `[hardened]` extras
- ANCHOR framework-adapter edge cases outside CI stub coverage

## Security Design Principles

ASHA follows these defaults:

- **Privacy-first**: PII masking enabled by default
- **Fail-safe**: Returns original or sanitized content on errors - never crashes the host app
- **No hidden downloads**: Rule-based / lite path requires no model downloads
- **Local processing**: Core security runs locally; no telemetry sent by default
- **Fail-closed on uncertain detections** when hardened paths are available (REVIEW / restrictive defaults rather than silent ALLOW)

## Best Practices for Users

- Install with `pip install asha-ai`; pin a tested version (e.g. `asha==0.4.2`) for production pilots
- Use `mode="strict"` for sensitive workloads
- Prefer `wrap_llm()` over `auto_patch()`
- Run `pip audit` regularly on your environment
- Do not log raw prompts containing PII in production
- Validate ASHA on your own data before compliance use
- Treat ANCHOR and experimental features as preview until 1.0
- Never commit `.env` or detector weights to the public repository
