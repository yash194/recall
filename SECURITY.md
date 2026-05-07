# Security Policy

## Reporting a vulnerability

If you discover a security vulnerability in Recall, please report it to
`security@recall.dev` (PGP key forthcoming).

Please do **not** file public GitHub issues for security vulnerabilities.

We will:
- Acknowledge receipt within 48 hours
- Provide a timeline for the fix within 7 days
- Credit you in the changelog (if you'd like) when the fix lands

## Disclosure window

90 days from initial report. If we haven't fixed it in 90 days, you're free
to disclose publicly.

## Supported versions

| Version | Supported |
|---|---|
| 0.4.x | yes |
| < 0.4 | no |

## Scope

In scope:
- The Python `recall` package (`src/recall/`)
- The HTTP server (`recall.server`) and MCP server (`recall.mcp_server`)
- The Docker image (`ghcr.io/yash194/recall`)
- The browser extension (`extensions/browser/`)

Out of scope:
- Third-party LLM providers (report to OpenAI/Anthropic/etc.)
- Vulnerabilities in dependencies (report upstream first)
- Issues in user-deployed self-hosted instances misconfigured for
  multi-tenant — please configure tenant isolation per `docs/DEPLOYMENT.md`

## What counts as a vulnerability

- **Critical**: cross-tenant data leak, RCE, auth bypass, hallucination-bound
  violation that returns false-positive support
- **High**: sensitive data exposure, audit-log tampering, persistent state
  corruption
- **Medium**: DoS, information disclosure that doesn't include user data
- **Low**: best-practice violations without exploitability

## Bounty

We don't run a paid bounty program at v0.4. If/when we do, it will be
announced here. We will publicly thank reporters in `CHANGELOG.md`.
