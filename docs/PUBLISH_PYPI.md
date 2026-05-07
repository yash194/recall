# Publishing Recall to PyPI — Complete Guide

> Step-by-step instructions to get `pip install recall` working for the world.
> Total time: ~30–45 minutes.
> Cost: $0.

---

## Table of contents

1. [What you'll end up with](#what-youll-end-up-with)
2. [Prerequisites](#prerequisites)
3. [Step 1 — Create PyPI accounts](#step-1--create-pypi-accounts)
4. [Step 2 — Check name availability](#step-2--check-name-availability)
5. [Step 3 — Verify your `pyproject.toml`](#step-3--verify-your-pyprojecttoml)
6. [Step 4 — Build the package locally](#step-4--build-the-package-locally)
7. [Step 5 — Test on TestPyPI (sandbox)](#step-5--test-on-testpypi-sandbox)
8. [Step 6 — Publish to real PyPI](#step-6--publish-to-real-pypi)
9. [Step 7 — Set up Trusted Publisher OIDC for automation](#step-7--set-up-trusted-publisher-oidc-for-automation)
10. [Step 8 — Tag a release and verify CI publishes](#step-8--tag-a-release-and-verify-ci-publishes)
11. [Versioning strategy](#versioning-strategy)
12. [Maintenance — releasing new versions](#maintenance--releasing-new-versions)
13. [Yanking a bad release](#yanking-a-bad-release)
14. [Troubleshooting](#troubleshooting)
15. [Quick checklist](#quick-checklist)

---

## What you'll end up with

After following this guide:

- `pip install recall` works on any machine in the world
- Tagging `v0.4.1` in git auto-publishes a new version (no manual upload)
- No long-lived API tokens stored anywhere (uses OIDC)
- Pre-1.0 release line: `0.4.0`, `0.4.1`, `0.5.0`, ... → eventually `1.0.0`

---

## Prerequisites

- A GitHub repo for Recall (public or private — both work)
- Python 3.10+ installed locally
- A 2FA app (Authy / 1Password / Google Authenticator / Bitwarden)
- Working internet connection for ~30 minutes

---

## Step 1 — Create PyPI accounts

### 1.1 Real PyPI account

1. Go to **https://pypi.org/account/register/**
2. Fill in: username (lowercase, e.g. `yash194`), email, password
3. Verify your email — click the link they send
4. Log in
5. Go to **Account Settings → Two-Factor Authentication**
6. PyPI **requires** 2FA for publishing. Set up TOTP (recommended) or a hardware key.
7. **Save the recovery codes somewhere safe** — if you lose your 2FA device and don't have these, you can't recover the account.

### 1.2 TestPyPI account (sandbox)

TestPyPI is a separate copy of PyPI used for testing. Packages there don't appear on real PyPI.

1. Go to **https://test.pypi.org/account/register/**
2. **It's a separate account** — you need to register again, even if you used the same email
3. Verify email, set up 2FA the same way
4. Save recovery codes

You'll use TestPyPI to rehearse your first upload before doing it for real.

---

## Step 2 — Check name availability

The name `recall` may already be taken on PyPI. Check **before** doing anything else.

```bash
curl -s -o /dev/null -w "PyPI 'recall': HTTP %{http_code}\n" https://pypi.org/pypi/recall/json
```

| HTTP code | Meaning |
|---|---|
| `200` | Name is taken — pick an alternative |
| `404` | Name is available — claim it |

Also check the GitHub username/org you want to use for namespacing (some find this useful in the package name):

```bash
curl -s -o /dev/null -w "GitHub user: HTTP %{http_code}\n" https://api.github.com/users/recall
```

### If `recall` is taken

PyPI names are first-come-first-served and can't be transferred without good cause. Alternatives in order of preference:

1. `recall-ai` — most common AI-product naming convention
2. `recall-memory` — descriptive
3. `recallkit` — implies a toolkit
4. `pyrecall` — `py` prefix is fine but slightly old-fashioned
5. `recall-mem` — short, distinctive
6. `recall-py`

Run the same `curl` for each candidate. Pick the first one that returns 404.

**Important**: pick a name you can also claim on **GitHub** and **npm** (for consistency). Run all three checks before deciding:

```bash
NAME="recall"     # change to whatever you're checking
curl -s -o /dev/null -w "PyPI:    %{http_code}\n" https://pypi.org/pypi/$NAME/json
curl -s -o /dev/null -w "npm:     %{http_code}\n" https://registry.npmjs.org/$NAME
curl -s -o /dev/null -w "GitHub:  %{http_code}\n" https://api.github.com/users/$NAME
```

All three should return `404`.

### If the name is available — should you claim it now?

**Yes, immediately.** Claiming a PyPI name is free, takes 10 minutes, and prevents typo-squatters from grabbing it before launch. You don't have to publish a real version on day one — just claim it with a placeholder.

To "claim" a name: upload at least one version of any package with that name. Even a stub `0.0.1` works.

---

## Step 3 — Verify your `pyproject.toml`

Open `pyproject.toml` and confirm these fields are correct. The current Recall `pyproject.toml` is already mostly right; double-check:

```toml
[project]
name = "recall"                                    # MUST match what you reserved on PyPI
version = "0.4.0"                                   # Semver, no leading 'v'
description = "Memory layer for AI agents — bounded hallucination, audit-grade, surgically forgettable"
readme = "README.md"                                # Must exist
requires-python = ">=3.10"
license = { text = "Apache-2.0" }
authors = [{ name = "Yash Aggarwal" }]
keywords = ["ai", "memory", "rag", "knowledge-graph", "llm-agents"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    "numpy>=1.24",
]

[project.urls]                                      # Add this if not present
Homepage = "https://github.com/yash194/recall"
Documentation = "https://github.com/yash194/recall/tree/main/docs"
Repository = "https://github.com/yash194/recall"
"Bug Tracker" = "https://github.com/yash194/recall/issues"
Changelog = "https://github.com/yash194/recall/blob/main/CHANGELOG.md"

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

**Things to verify**:
- `name` is the exact PyPI name you claimed
- `version` follows semver — no `0.4.0-alpha` or `v0.4.0`, just `0.4.0`
- `readme = "README.md"` — and that file exists
- `license = { text = "Apache-2.0" }` — uses the SPDX identifier
- `[tool.setuptools.packages.find]` points at your source layout (`src/`)

---

## Step 4 — Build the package locally

```bash
cd /Users/yashaggarwal/Downloads/PageIndex/recall

# Install build tools
pip install --upgrade pip build twine

# Clean old artifacts (important — stale dist/ files cause weird upload errors)
rm -rf dist/ build/ src/*.egg-info/

# Build sdist + wheel
python -m build
```

You should see output like:

```
Successfully built recall-0.4.0.tar.gz and recall-0.4.0-py3-none-any.whl
```

Verify the artifacts:

```bash
ls -la dist/
# recall-0.4.0-py3-none-any.whl
# recall-0.4.0.tar.gz
```

Quick sanity-check the contents of the wheel:

```bash
python -m zipfile -l dist/recall-0.4.0-py3-none-any.whl | head -30
```

You should see all your `recall/*.py` files listed. If you see anything weird (like `__pycache__/` directories or your tests folder), that's a packaging bug — fix `[tool.setuptools.packages.find]` in `pyproject.toml`.

Validate metadata:

```bash
twine check dist/*
# Should print:
# Checking dist/recall-0.4.0-py3-none-any.whl: PASSED
# Checking dist/recall-0.4.0.tar.gz: PASSED
```

If `twine check` flags issues, fix them before uploading. Common issues:
- README rendering errors → check the README.md preview at `https://pypi.org/`
- Long description format → set `readme = "README.md"` (not just a string)

---

## Step 5 — Test on TestPyPI (sandbox)

**Always test on TestPyPI first** before real PyPI. TestPyPI deletions are easier and you don't burn a real version number.

### 5.1 Get a TestPyPI API token

1. Log in at **https://test.pypi.org/manage/account/**
2. Scroll down to **API tokens**
3. Click **Add API token**
4. Token name: `recall-first-upload`
5. Scope: **Entire account** (only for the very first upload — see security note below)
6. Click **Add token**
7. **Copy the token immediately** (starts with `pypi-AgENdGV...`). You won't see it again.
8. Save it temporarily in a password manager

### 5.2 Upload to TestPyPI

```bash
twine upload --repository testpypi dist/*
```

When prompted:
- Username: `__token__` (literally — two underscores)
- Password: paste the token you just created (starts with `pypi-...`)

If it works, you'll see:

```
Uploading recall-0.4.0-py3-none-any.whl
100%
Uploading recall-0.4.0.tar.gz
100%

View at:
https://test.pypi.org/project/recall/0.4.0/
```

Open the URL — verify the README renders correctly, classifiers show up, etc.

### 5.3 Install from TestPyPI to verify

In a fresh virtualenv:

```bash
python -m venv /tmp/testenv
source /tmp/testenv/bin/activate

# Install from TestPyPI, but pull dependencies from real PyPI
# (TestPyPI doesn't have all packages mirrored)
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            recall

# Smoke test
python -c "from recall import Memory; m = Memory(tenant='test'); print('OK')"
```

If that prints `OK`, your package works. Deactivate:

```bash
deactivate
rm -rf /tmp/testenv
```

### 5.4 If something goes wrong

You can re-upload the same version to TestPyPI by deleting the project from your TestPyPI account first. To do this:

1. Go to https://test.pypi.org/manage/project/recall/
2. **Settings** → scroll down → **Delete project**
3. Type the project name to confirm

Then rebuild and re-upload. Real PyPI does **not** allow this — you'd need to bump the version. Hence why we test here first.

### 5.5 Revoke the wide-scope token (security)

Now that the upload worked once, narrow the token scope:

1. Go to https://test.pypi.org/manage/account/token/
2. Revoke the `recall-first-upload` token
3. Create a new token scoped to **the `recall` project only**
4. Save the new token (you'll only need it for manual uploads — automation uses OIDC, no tokens)

---

## Step 6 — Publish to real PyPI

Once TestPyPI works end-to-end, do the same on real PyPI.

### 6.1 Get a real PyPI API token

1. Log in at **https://pypi.org/manage/account/**
2. **API tokens** → **Add API token**
3. Token name: `recall-first-upload`
4. Scope: **Entire account** (we'll narrow it after the first upload claims the project)
5. Copy the token (starts with `pypi-...`)

### 6.2 Upload to real PyPI

Make sure your `dist/` is fresh (rebuild if you changed anything since TestPyPI):

```bash
rm -rf dist/ build/ src/*.egg-info/
python -m build
twine check dist/*
twine upload dist/*
```

When prompted:
- Username: `__token__`
- Password: paste the real PyPI token

You should see:

```
Uploading recall-0.4.0-py3-none-any.whl
100%
Uploading recall-0.4.0.tar.gz
100%

View at:
https://pypi.org/project/recall/0.4.0/
```

The package is now public. **Anyone in the world can run `pip install recall`.**

### 6.3 Verify

In a fresh shell on a different machine if possible:

```bash
pip install recall
python -c "from recall import Memory; print(Memory)"
```

### 6.4 Narrow the token scope

The `Entire account` token is too powerful to leave around. Now that the project exists:

1. https://pypi.org/manage/account/token/
2. Revoke `recall-first-upload`
3. Create a new token scoped to **just the `recall` project** (you'll see it in the dropdown now)
4. Use that for any future manual uploads

But you actually shouldn't need manual uploads after this — Step 7 sets up automation.

---

## Step 7 — Set up Trusted Publisher OIDC for automation

Manual uploads with API tokens have two problems:
1. The token can leak (committed by accident, screenshot, etc.)
2. You have to manually run `twine upload` every release

**Trusted Publisher OIDC** solves both: GitHub Actions authenticates to PyPI using a short-lived OIDC token, no API key needed.

### 7.1 Configure on PyPI

1. Go to **https://pypi.org/manage/project/recall/settings/publishing/**
2. Click **Add a new pending publisher** (or **Add a new trusted publisher** if the project already exists)
3. Fill in:
   - **PyPI Project Name**: `recall`
   - **Owner**: your GitHub username or org (e.g. `yash194`)
   - **Repository name**: `recall` (the GitHub repo name)
   - **Workflow name**: `release.yml` (the file name in `.github/workflows/`)
   - **Environment name**: leave blank (we don't use a deployment environment)
4. Click **Add**

That's it on the PyPI side.

### 7.2 Verify your release workflow

Open `.github/workflows/release.yml`. The relevant job should look like this (already in place):

```yaml
jobs:
  pypi:
    name: build and publish to PyPI (Trusted Publisher OIDC)
    runs-on: ubuntu-latest
    permissions:
      id-token: write   # required for PyPI Trusted Publisher
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Build distribution
        run: |
          python -m pip install --upgrade pip build
          python -m build
      - name: Publish to PyPI (OIDC)
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          skip-existing: true
```

The key part is `permissions: id-token: write`. This lets GitHub mint the OIDC token PyPI verifies.

**No `password:` field. No `secrets.PYPI_TOKEN`.** That's the whole point.

### 7.3 Repeat for TestPyPI (optional but recommended)

Add a second pending publisher at https://test.pypi.org/manage/account/publishing/ pointing at the same workflow. Set up a second job in `release.yml` for TestPyPI if you want pre-release dry-runs:

```yaml
  testpypi:
    name: publish to TestPyPI
    runs-on: ubuntu-latest
    if: github.ref == 'refs/tags/v0.0.0-test'   # only on test tags
    permissions:
      id-token: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install build && python -m build
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          repository-url: https://test.pypi.org/legacy/
          skip-existing: true
```

---

## Step 8 — Tag a release and verify CI publishes

The first time you do this, do it for `v0.4.1` (a tiny patch above what you uploaded manually) so you can confirm automation works.

### 8.1 Bump the version

Edit `pyproject.toml`:

```toml
version = "0.4.1"
```

Update `CHANGELOG.md`:

```markdown
## [0.4.1] - 2026-05-08

### Added
- (Document what changed)
```

Commit:

```bash
git add pyproject.toml CHANGELOG.md
git commit -s -m "chore: release v0.4.1"
git push
```

### 8.2 Tag and push

```bash
git tag v0.4.1
git push --tags
```

### 8.3 Watch CI

Go to https://github.com/yash194/recall/actions

You should see the `release` workflow run. The `pypi` job:
1. Checks out the code
2. Builds the wheel and sdist
3. Authenticates to PyPI via OIDC (no token needed)
4. Uploads

Total time: ~2 minutes.

### 8.4 Verify

```bash
pip install --upgrade recall
pip show recall | grep Version
# Version: 0.4.1
```

You're done. Future releases are just:

```bash
git tag v0.5.0
git push --tags
```

The CI handles the rest.

---

## Versioning strategy

Recall follows **semver** (semantic versioning):

| Version | When |
|---|---|
| **0.x.y** (current) | Pre-1.0. Breaking changes allowed in minor (`0.4 → 0.5`) bumps. |
| **1.0.0** | First stable. After this, breaking changes only on major bumps. |
| **MAJOR.MINOR.PATCH** | `1.4.2`. Breaking → MAJOR. New features → MINOR. Bug fixes → PATCH. |

**Tag format**: always `v` prefix in the git tag, never in `pyproject.toml`.

```
git tag v0.4.1   ← yes
version = "0.4.1" in pyproject.toml   ← yes (no v)
```

Pre-release suffixes (rare, but supported):

```
0.5.0a1     # alpha 1
0.5.0b2     # beta 2
0.5.0rc1    # release candidate 1
```

PyPI honors PEP 440 versioning. Pre-releases don't install by default; users must pass `--pre`.

---

## Maintenance — releasing new versions

The repeat-pattern from this point forward:

1. **Make changes**, write tests, get them merged to `main`
2. **Decide a version number** (patch / minor / major)
3. **Update `pyproject.toml` `version` and `CHANGELOG.md`**
4. **Commit + push** to `main`
5. **Tag**: `git tag v0.X.Y && git push --tags`
6. CI publishes automatically.

There is no other manual work.

### Pre-release rehearsal

If you want to dry-run a release without burning a version number:

1. Set up the TestPyPI publisher (Step 7.3)
2. Create a tag like `v0.5.0-test1` and push
3. The TestPyPI workflow runs; verify install
4. Once happy, bump to `v0.5.0` and push the real tag

---

## Yanking a bad release

If you publish a broken version (rare but happens), you can **yank** it. Yanking doesn't delete the file — `pip install recall==0.4.5` still works for users with a pinned version — but new installs and `pip install recall` (unpinned) will skip it.

1. Go to https://pypi.org/manage/project/recall/release/0.4.5/
2. Click **Options → Yank**
3. Confirm

Then publish a fixed version (`0.4.6`).

**Don't yank as a "delete"** — yanked versions are still public. If you accidentally uploaded sensitive data, contact PyPI support immediately to request actual deletion.

---

## Troubleshooting

### `400 Bad Request: File already exists`

You're trying to upload a version that's already on PyPI. Bump the version number in `pyproject.toml` and rebuild.

### `403 Forbidden`

- Wrong API token, or token is for wrong project scope
- Token has been revoked — generate a new one
- Trusted Publisher OIDC misconfigured — check repo name / workflow name match exactly

### `403 Invalid or non-existent authentication information`

The token format is wrong. Username must be **literally** `__token__` (two leading + two trailing underscores). The password is the entire token string starting with `pypi-`.

### `README.md doesn't render on PyPI`

Make sure `readme = "README.md"` (not `readme = {file = "README.md"}`). Check `twine check dist/*` for warnings.

### `pip install recall` installs an old version

PyPI caches aggressively. Force-refresh:

```bash
pip install --upgrade --force-reinstall --no-cache-dir recall
```

If that still pulls an old version, your local pip might be configured to use a mirror or an internal index. Check `pip config list`.

### CI fails with "no permission to publish"

Trusted Publisher mismatch. Common causes:
- Repo was renamed → update the trusted publisher entry on PyPI
- Workflow file renamed → update the trusted publisher entry
- Workflow runs from a fork → trusted publishers don't work from forks; you must run from the canonical repo

### `python -m build` fails with "package not found"

Your `[tool.setuptools.packages.find]` doesn't match your layout. Recall uses `src/` layout, so:

```toml
[tool.setuptools.packages.find]
where = ["src"]
```

If your code is at the repo root instead, omit the `where` line.

### Name conflict at upload time

If `recall` was free when you checked but taken at upload, someone (or a typo-squatter bot) registered it between your check and your upload. PyPI now auto-flags typo-squats but doesn't prevent legitimate registrations.

Pick a fallback name (`recall-ai`, `recall-mem`) and update `pyproject.toml`'s `name` field. Keep your import path (`import recall`) the same — only the *PyPI distribution name* changes:

```toml
[project]
name = "recall-ai"   # PyPI distribution name

# users still do: from recall import Memory
```

---

## Quick checklist

```
PyPI launch
───────────
□  Create pypi.org account
□  Set up 2FA, save recovery codes
□  Create test.pypi.org account, set up 2FA
□  Check name availability:  curl https://pypi.org/pypi/recall/json
□  Verify pyproject.toml: name, version, readme, license, classifiers
□  Build:    rm -rf dist/ build/ && python -m build
□  Validate: twine check dist/*
□  TestPyPI: twine upload --repository testpypi dist/*
□  Verify install from TestPyPI in a fresh venv
□  Real PyPI: twine upload dist/*
□  Verify install:  pip install recall
□  Set up Trusted Publisher OIDC at pypi.org/manage/.../publishing/
□  Bump version in pyproject.toml
□  Tag and push:   git tag v0.4.1 && git push --tags
□  Confirm CI auto-publishes the new version
□  Revoke the broad API token; keep only project-scoped tokens
□  Document the release process for future maintainers (this file)
```

---

## Related

- [PyPI Trusted Publishers documentation](https://docs.pypi.org/trusted-publishers/)
- [`pypa/gh-action-pypi-publish`](https://github.com/pypa/gh-action-pypi-publish) — the GitHub Action used in `release.yml`
- [PEP 440 — Version specifiers](https://peps.python.org/pep-0440/)
- [Recall's `release.yml` workflow](../.github/workflows/release.yml)
- [PyPI's public help](https://pypi.org/help/)
