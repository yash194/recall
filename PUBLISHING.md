# Publishing Recall

> Step-by-step guide for shipping a new release to PyPI, npm, and GitHub.
> Last verified: 2026-05-08, version 0.1.0.

## What's in the box

| Artifact | Location | Distributed via |
|---|---|---|
| Python package | `pyproject.toml` + `src/recall/` | PyPI (`pip install recall`) |
| npm wrapper | `npm/package.json` + `npm/bin/recall-mcp.js` | npm (`npm install -g recall-mcp`) |
| Source repo | `.` (everything) | GitHub (`yash194/recall`) |

Python is the canonical implementation. The npm package is a thin Node
shim that runs the Python MCP server via `uvx` / `pipx`.

---

## One-time setup (you, the publisher)

### 1. Authenticate with GitHub
```bash
gh auth login
# Select "GitHub.com" → "HTTPS" → "Login with a web browser"
```

### 2. PyPI account + API token
1. Create account at <https://pypi.org/account/register/>.
2. Verify email.
3. Optional but recommended: register `recall` package name first via
   <https://test.pypi.org> to make sure no one else grabs it.
4. Settings → API tokens → "Add API token" — scope: "Entire account"
   for the first publish; after publishing, narrow to "recall" project.
5. Save the token to `~/.pypirc`:
   ```ini
   [pypi]
     username = __token__
     password = pypi-<paste-token-here>
   ```

### 3. npm account + token
```bash
npm login
# Or, if you have 2FA: npm login --auth-type=web
```
Confirm: `npm whoami`.

### 4. Install build tools (one-time)
```bash
pip install --upgrade build twine
# `build` produces wheel + sdist
# `twine` uploads to PyPI
```

---

## Per-release workflow

### Step 1 — bump version everywhere
Edit these files together so PyPI and npm stay in sync:

- `pyproject.toml`: `version = "X.Y.Z"`
- `npm/package.json`: `"version": "X.Y.Z"`
- `CHANGELOG.md`: add a `## vX.Y.Z — YYYY-MM-DD` section at the top

### Step 2 — sanity-check
```bash
# All tests pass
pip install -e '.[dev]'
pytest -x -q

# Build is clean
rm -rf dist build *.egg-info
python -m build
ls dist/  # → recall-X.Y.Z.tar.gz, recall-X.Y.Z-py3-none-any.whl

# Verify wheel contents
unzip -l dist/recall-X.Y.Z-py3-none-any.whl | head -30

# Verify npm tarball
( cd npm && npm pack --dry-run )
```

### Step 3 — push to GitHub
```bash
git add -A
git commit -m "release: vX.Y.Z"
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin main --tags
gh release create vX.Y.Z --title "vX.Y.Z" --notes-from-tag
```

### Step 4 — publish to PyPI
```bash
# (optional) test on TestPyPI first:
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ typed-recall

# Real publish:
twine upload dist/*
```
Wait ~1 minute, then verify: `pip install --upgrade typed-recall`.

### Step 5 — publish to npm
```bash
cd npm
npm publish
cd ..
```
Verify: `npx -y recall-mcp@latest --help` (will exit 0 after spawning Python).

### Step 6 — smoke-test the released artifacts
```bash
# PyPI
pip install --upgrade 'typed-recall[mcp,embed-bge,llm-openai]'
python -c "from recall import Memory; print(Memory(tenant='smoke').stats())"
recall me add 'release smoke test memory'
recall me ask 'release smoke test'

# npm
npx -y recall-mcp 2>&1 | head -3   # should print the launching hint and start the MCP server

# MCP integration
claude mcp add recall-test -- npx -y recall-mcp
# In a fresh `claude` session, call mcp__recall-test__stats
claude mcp remove recall-test
```

---

## First-time bootstrap (this release, v0.1.0)

You haven't shipped before. Do this in order:

```bash
# 1. Create the GitHub repo
gh repo create yash194/recall --public --source=. --remote=origin \
    --description "Memory layer for AI agents — typed-edge graph, bounded hallucination, audit-grade, surgically forgettable. Local SQLite + MCP server."

# 2. First push
git push -u origin main

# 3. Tag the release
git tag -a v0.1.0 -m "v0.1.0 — first public release"
git push origin v0.1.0

# 4. Build + upload Python
python -m build
twine upload dist/*

# 5. Publish npm
cd npm && npm publish && cd ..

# 6. Cut a GitHub release
gh release create v0.1.0 \
    --title "v0.1.0 — first public release" \
    --notes-file <(sed -n '/^## v0.1.0/,/^---$/p' CHANGELOG.md)
```

---

## When something goes wrong

| Problem | Fix |
|---|---|
| `twine` says version already exists | You can't overwrite a PyPI version. Bump to `0.1.1` and re-upload. |
| npm package name `recall` is taken | We're using `recall-mcp` for npm because plain `recall` is a popular npm name (UI library). Don't try to rename. |
| `gh repo create` says repo exists | Either delete it (`gh repo delete yash194/recall`) or skip create and just add the remote: `git remote add origin git@github.com:yash194/recall.git`. |
| npm `403 Forbidden` on publish | Run `npm login` again. If still failing, check `npm whoami` matches the package's `author` / org. |
| PyPI `400 The description failed to render` | The README has a syntax error in its markdown. Fix and re-build. |
| Wheel is huge (>10MB) | `MANIFEST.in` is including too much. Check `prune tests` / `prune benchmarks` are listed. |

---

## Versioning policy

- **0.1.x** — alpha. Public API can change between patches without warning.
  Suitable for: research, experimentation, MCP-only use.
- **0.x** (general) — minor versions can break API. Document in CHANGELOG.
- **1.0** — first SLA-able release. Stable API for at least 6 months.
- All breaking changes documented in CHANGELOG with migration notes.

---

## What NOT to publish

- **Never** publish a version with failing tests. Pre-commit hook
  enforces 154 unit tests must pass.
- **Never** publish credentials. `dist/`, `build/`, `*.egg-info` are
  in `.gitignore`; verify they're not tracked.
- **Never** publish private benchmark results (e.g., a customer's data
  ingested into `~/.recall/`). Only synthetic + public datasets.
