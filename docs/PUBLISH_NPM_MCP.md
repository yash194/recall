# Publishing the Recall MCP Server to npm — Complete Guide

> Step-by-step instructions to ship `npx @recall/mcp` (or similar) so Claude
> Desktop / Cursor / Cline / Codex users can install Recall as an MCP server
> in one command without needing Python.
>
> Total time: ~30 minutes.
> Cost: $0.

---

## Table of contents

1. [What you'll end up with](#what-youll-end-up-with)
2. [Should you ship this at all?](#should-you-ship-this-at-all)
3. [Prerequisites](#prerequisites)
4. [Step 1 — Create an npm account](#step-1--create-an-npm-account)
5. [Step 2 — Decide on the package name](#step-2--decide-on-the-package-name)
6. [Step 3 — Set up the npm package directory](#step-3--set-up-the-npm-package-directory)
7. [Step 4 — Write the wrapper script](#step-4--write-the-wrapper-script)
8. [Step 5 — Test locally with `npm link`](#step-5--test-locally-with-npm-link)
9. [Step 6 — First publish to npm](#step-6--first-publish-to-npm)
10. [Step 7 — Verify install via `npx`](#step-7--verify-install-via-npx)
11. [Step 8 — Configure Claude Desktop / Cursor / Codex](#step-8--configure-claude-desktop--cursor--codex)
12. [Step 9 — Automate releases via GitHub Actions](#step-9--automate-releases-via-github-actions)
13. [Versioning strategy](#versioning-strategy)
14. [Maintenance — new releases](#maintenance--new-releases)
15. [Unpublishing & deprecating](#unpublishing--deprecating)
16. [Troubleshooting](#troubleshooting)
17. [Quick checklist](#quick-checklist)

---

## What you'll end up with

After following this guide:

- `npx -y @recall-ai/mcp` runs the Recall MCP server in any terminal
- Claude Desktop / Cursor / Cline / Codex / Continue users can add Recall via:

  ```json
  {
    "mcpServers": {
      "recall": {
        "command": "npx",
        "args": ["-y", "@recall-ai/mcp"]
      }
    }
  }
  ```

- New versions auto-publish to npm when you tag a release in git
- Provenance attestation proves the package was built from your CI

The npm package is a **thin wrapper** around the Python MCP server — it calls
`uvx --from recall recall-mcp` under the hood. So users get the npm install
UX (`npx`) without you maintaining a separate code path.

---

## Should you ship this at all?

Honestly: **maybe not at v0.4.**

Reasons to ship npm:
- Reach Node.js developers who don't have Python installed
- Lets users install via `npx` (single command, no environment management)
- Adds visibility on the npm registry (separate audience from PyPI)

Reasons to skip:
- `uvx --from recall recall-mcp` already works for the same use case
- Most MCP-aware clients (Claude Desktop, Cursor, Cline) already prefer `uvx`
- Adds another release path to maintain
- Many MCP server authors haven't bothered with npm

**Recommendation**: ship PyPI first. Only add npm if you get real signal that users want it (search for "recall npm" in issues / Discord / Twitter).

If you decide to ship it now, this guide gets you there in ~30 minutes.

---

## Prerequisites

- Node.js 18+ and npm 9+ installed locally:
  ```bash
  brew install node            # macOS
  # or use nvm: https://github.com/nvm-sh/nvm
  node -v   # v18+
  npm -v    # 9+
  ```
- A 2FA app (Authy / 1Password / Google Authenticator)
- A GitHub repo (the npm package can live in the same repo as the Python source, or in its own repo — this guide uses a subdirectory)
- The Python `recall` package already published to PyPI (per `PUBLISH_PYPI.md`)

---

## Step 1 — Create an npm account

1. Go to **https://www.npmjs.com/signup**
2. Choose a username (lowercase, e.g. `yash194`). This becomes your scope (`@yash194/...`)
3. Verify your email
4. **Enable 2FA**: `Account Settings → Two-Factor Authentication`
   - npm requires 2FA for publishing as of 2024
   - Choose TOTP (recommended) or a hardware key
5. **Save the recovery codes**

(Optional but recommended) — **Create an organization** for cleaner branding:

1. https://www.npmjs.com/org/create
2. Org name: `recall-ai` (or whatever you reserved)
3. Free tier is fine
4. Now you can publish as `@recall-ai/mcp` instead of `@yash194/recall-mcp`

---

## Step 2 — Decide on the package name

### Option A — unscoped: `recall-mcp`

Check availability:
```bash
npm view recall-mcp 2>&1 | head -1
# "npm error 404 Not Found" = available
# anything else = taken
```

Unscoped names are global and first-come-first-served. Risky and competitive.

### Option B — user-scoped: `@yash194/recall-mcp`

Always available because it's namespaced to you. Unique disadvantage: looks like a personal package, not a project package.

### Option C — org-scoped: `@recall-ai/mcp`

Requires creating the npm org (free, instant). Looks professional and distinct.

**Recommendation**: **Option C**. Create `@recall-ai` org, publish `@recall-ai/mcp`, `@recall-ai/sdk` (later), etc. Same naming you'd use on PyPI alternatives if `recall` was taken.

Throughout this guide we'll use `@recall-ai/mcp` as the example. Substitute your chosen name.

---

## Step 3 — Set up the npm package directory

Decide where the npm wrapper lives. Two options:

**Option A** (recommended) — same repo as Python source, in a subdirectory:

```
recall/                        ← your existing repo
├── src/recall/                ← Python source
├── pyproject.toml             ← Python package
├── npm/                       ← NEW
│   ├── package.json
│   ├── bin/
│   │   └── recall-mcp.js
│   └── README.md
└── ...
```

**Option B** — separate repo `recall-mcp-npm/`:

```
recall-mcp-npm/
├── package.json
├── bin/
│   └── recall-mcp.js
└── README.md
```

Option A is simpler for now. Use it.

### Create the directory

```bash
cd /Users/yashaggarwal/Downloads/PageIndex/recall
mkdir -p npm/bin
cd npm
```

### Create `package.json`

```bash
npm init -y
```

This generates a stub. Replace its contents with:

```json
{
  "name": "@recall-ai/mcp",
  "version": "0.4.0",
  "description": "MCP server for Recall — typed-edge memory for AI agents. Ships as a thin wrapper around the Python recall-mcp executable.",
  "bin": {
    "recall-mcp": "./bin/recall-mcp.js"
  },
  "files": [
    "bin/",
    "README.md",
    "LICENSE"
  ],
  "keywords": [
    "mcp",
    "model-context-protocol",
    "ai",
    "memory",
    "agent",
    "claude",
    "cursor",
    "codex",
    "cline",
    "continue"
  ],
  "author": "Yash Aggarwal",
  "license": "Apache-2.0",
  "homepage": "https://github.com/yash194/recall",
  "repository": {
    "type": "git",
    "url": "https://github.com/yash194/recall.git",
    "directory": "npm"
  },
  "bugs": {
    "url": "https://github.com/yash194/recall/issues"
  },
  "engines": {
    "node": ">=18"
  },
  "publishConfig": {
    "access": "public"
  }
}
```

Key fields:
- `name`: `@recall-ai/mcp` — the scoped name
- `bin.recall-mcp`: tells npm to install `recall-mcp` as an executable
- `files`: only these files get packed into the published tarball
- `engines.node`: minimum Node version
- `publishConfig.access`: scoped packages default to `restricted`; we want public

### Copy the LICENSE

```bash
cp ../LICENSE LICENSE
```

---

## Step 4 — Write the wrapper script

Create `bin/recall-mcp.js`:

```javascript
#!/usr/bin/env node
/**
 * @recall-ai/mcp — npm wrapper around the Recall MCP server.
 *
 * Spawns `uvx --from recall recall-mcp` (the Python MCP server) and forwards
 * stdio. This means the npm package only needs Node ≥18 — uvx handles fetching
 * Python and the recall package transparently.
 *
 * Override the spawn command via env vars:
 *   RECALL_MCP_COMMAND   — full command (default: 'uvx')
 *   RECALL_MCP_ARGS      — comma-separated args (default: '--from,recall,recall-mcp')
 *
 * If uvx is not installed, prints a helpful message.
 */

const { spawn } = require('child_process');
const { existsSync } = require('fs');

function findExecutable(name) {
  const paths = (process.env.PATH || '').split(require('path').delimiter);
  const ext = process.platform === 'win32' ? '.exe' : '';
  for (const p of paths) {
    const full = require('path').join(p, name + ext);
    if (existsSync(full)) return full;
  }
  return null;
}

function main() {
  const cmd  = process.env.RECALL_MCP_COMMAND || 'uvx';
  const args = (process.env.RECALL_MCP_ARGS
    ? process.env.RECALL_MCP_ARGS.split(',')
    : ['--from', 'recall', 'recall-mcp']);

  // Helpful preflight check if user is using the default uvx path.
  if (cmd === 'uvx' && !findExecutable('uvx')) {
    process.stderr.write(
      'recall-mcp: uvx is not installed.\n' +
      '\n' +
      'Install uv (which provides uvx) — one of:\n' +
      '  curl -LsSf https://astral.sh/uv/install.sh | sh\n' +
      '  brew install uv\n' +
      '  pipx install uv\n' +
      '\n' +
      'Or override the spawn command:\n' +
      '  RECALL_MCP_COMMAND=python RECALL_MCP_ARGS=-m,recall.mcp_server npx -y @recall-ai/mcp\n'
    );
    process.exit(127);
  }

  const child = spawn(cmd, args, {
    stdio: 'inherit',
    env: process.env,
  });

  child.on('error', (err) => {
    process.stderr.write(`recall-mcp: failed to spawn ${cmd}: ${err.message}\n`);
    process.exit(127);
  });

  child.on('exit', (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal);
    } else {
      process.exit(code ?? 0);
    }
  });

  // Forward signals so Ctrl-C cleanly terminates the Python child.
  ['SIGINT', 'SIGTERM', 'SIGHUP'].forEach(sig => {
    process.on(sig, () => child.kill(sig));
  });
}

main();
```

Make it executable:

```bash
chmod +x bin/recall-mcp.js
```

### Create the npm package README

```bash
cat > README.md <<'EOF'
# @recall-ai/mcp

MCP (Model Context Protocol) server for [Recall](https://github.com/yash194/recall) —
typed-edge memory substrate for AI agents.

This npm package is a thin Node wrapper. It spawns the Python MCP server via
`uvx --from recall recall-mcp` so users don't need to manage Python
environments themselves.

## Install

In any MCP-aware client (Claude Desktop, Cursor, Cline, Codex, Continue, Zed, Windsurf):

```json
{
  "mcpServers": {
    "recall": {
      "command": "npx",
      "args": ["-y", "@recall-ai/mcp"]
    }
  }
}
```

For Claude Code:

```bash
claude mcp add recall -- npx -y @recall-ai/mcp
```

## Requirements

- **Node.js ≥ 18**
- **uv** (https://docs.astral.sh/uv/) — provides `uvx`. Install one of:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # or
  brew install uv
  ```

The first invocation downloads the Recall Python package automatically.

## Tools exposed

The MCP server registers 8 tools:

| Tool | What it does |
|---|---|
| `add_memory` | Store a memory with quality gating, dedup, edge induction |
| `search_memory` | Connected reasoning subgraph for a query |
| `bounded_answer` | Generate an answer bounded by structural support |
| `forget` | Surgical delete with audit log |
| `audit` | Provenance trail for any operation |
| `graph_health` | Spectral / topology / curvature health metrics |
| `consolidate` | Run sleep-time consolidation |
| `stats` | Active nodes / edges / audit count |

## Configuration

Override the spawn target via environment variables:

| Var | Default | Use when |
|---|---|---|
| `RECALL_MCP_COMMAND` | `uvx` | You don't have uv; use `python` instead |
| `RECALL_MCP_ARGS` | `--from,recall,recall-mcp` | You're running from source: `-m,recall.mcp_server` |
| `RECALL_DB_DIR` | `~/.recall` | Where SQLite databases are stored |

## Source

Source code, documentation, and the Python implementation:
**https://github.com/yash194/recall**

## License

Apache-2.0.
EOF
```

### Verify the package contents

```bash
npm pack --dry-run
```

You should see something like:

```
npm notice 📦  @recall-ai/mcp@0.4.0
npm notice === Tarball Contents ===
npm notice 1.2kB LICENSE
npm notice 2.0kB README.md
npm notice 256B  package.json
npm notice 1.8kB bin/recall-mcp.js
```

If you see `node_modules/` or other junk, add to `.npmignore` or list only necessary paths in `package.json`'s `files` array.

---

## Step 5 — Test locally with `npm link`

Before publishing, verify the package works locally:

```bash
cd /Users/yashaggarwal/Downloads/PageIndex/recall/npm
npm link
```

This creates a symlink so `recall-mcp` is callable globally. Test:

```bash
which recall-mcp
# /usr/local/bin/recall-mcp (or similar)

# Try running it. If `uv` is installed, this should start the MCP server:
recall-mcp <<< '{}'
# (it'll wait for JSON-RPC input — Ctrl-C to exit)
```

Even better — test via the official MCP Inspector:

```bash
npx @modelcontextprotocol/inspector recall-mcp
```

The inspector opens a UI listing the 8 tools. If they all appear and `add_memory` works in a test call, your package is ready.

When done testing:

```bash
npm unlink -g @recall-ai/mcp
```

---

## Step 6 — First publish to npm

### 6.1 Log in

```bash
npm login
```

It opens a browser to npmjs.com. Authenticate (with 2FA). Confirm:

```bash
npm whoami
# yash194
```

### 6.2 Final pre-publish checks

```bash
# Verify your version isn't already taken
npm view @recall-ai/mcp version 2>&1 | head -3
# Should say "npm error 404 Not Found" the first time

# Lint package.json
npm pkg fix
```

### 6.3 Publish

```bash
cd /Users/yashaggarwal/Downloads/PageIndex/recall/npm
npm publish --access public
```

You'll be asked for your 2FA OTP code. Enter it.

If successful:

```
+ @recall-ai/mcp@0.4.0
```

Verify:

```bash
npm view @recall-ai/mcp
```

You should see your package metadata: version, description, license, repository URL, etc.

The package page is now live at:

```
https://www.npmjs.com/package/@recall-ai/mcp
```

---

## Step 7 — Verify install via `npx`

In a fresh shell:

```bash
# Force-reinstall so we don't hit npx's cache
rm -rf ~/.npm/_npx

# Should download the package and run it
npx -y @recall-ai/mcp <<< '{}'
# (it waits for input)
# Ctrl-C to exit
```

If that works, anyone in the world can now add Recall as an MCP server with:

```bash
claude mcp add recall -- npx -y @recall-ai/mcp
```

---

## Step 8 — Configure Claude Desktop / Cursor / Codex

### Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "recall": {
      "command": "npx",
      "args": ["-y", "@recall-ai/mcp"]
    }
  }
}
```

Restart Claude Desktop. In a chat, type `/mcp` and you should see `recall` listed.

### Cursor

`Settings → MCP → Add new MCP server`:
- **Name**: recall
- **Command**: npx
- **Args**: `-y @recall-ai/mcp`

### Codex CLI

Edit `~/.codex/config.toml`:

```toml
[mcp_servers.recall]
command = "npx"
args = ["-y", "@recall-ai/mcp"]
```

### Cline / Continue / Zed / Windsurf

Same pattern — each tool has a settings UI for adding MCP servers. Use:
- command: `npx`
- args: `-y @recall-ai/mcp`

### Verify cross-tool memory

This is the demo from `docs/CUSTOMER_JOURNEY.md` persona 2:

1. In Claude Desktop, ask:
   > "Use the recall add_memory tool to remember: We use Redis Streams as the queue."

2. Close Claude. Open Cursor.

3. Ask Cursor:
   > "Use the recall search_memory tool to find what queue tech we use."

4. Cursor returns the Claude-stored memory. Same database, two tools.

---

## Step 9 — Automate releases via GitHub Actions

Manual `npm publish` is fine for first-time setup; for future releases, automate.

### 9.1 Generate an npm token

1. https://www.npmjs.com/settings/yash194/tokens → **Generate New Token (Granular)**
2. Token name: `recall-mcp-ci`
3. Expiration: 1 year (renew yearly)
4. Permissions: **read and write**
5. Packages and scopes: scope to `@recall-ai/mcp` (or your package)
6. Click **Generate Token**
7. Copy the token immediately

### 9.2 Add to GitHub Secrets

1. Repo settings → **Secrets and variables → Actions** → **New repository secret**
2. Name: `NPM_TOKEN`
3. Value: paste the token

### 9.3 Update `.github/workflows/release.yml`

Append a new job to publish to npm on tagged releases:

```yaml
  npm:
    name: publish to npm
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write   # for npm provenance
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
          registry-url: https://registry.npmjs.org/
      - name: Sync npm package version with git tag
        run: |
          cd npm
          VERSION="${GITHUB_REF#refs/tags/v}"
          npm version "$VERSION" --no-git-tag-version --allow-same-version
      - name: Publish with provenance
        working-directory: npm
        run: npm publish --access public --provenance
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
```

What this does:
- Checks out the repo on every `v*.*.*` tag
- Sets up Node 20
- Updates `npm/package.json` `version` to match the git tag (no commit, just for the publish)
- Publishes with **provenance attestation** — npm verifies the package was built from this exact GitHub Actions run

### 9.4 Test the automation

1. Bump versions:
   - `pyproject.toml` → `0.4.1`
   - `npm/package.json` → `0.4.1` (must match)
2. Commit:
   ```bash
   git add pyproject.toml npm/package.json CHANGELOG.md
   git commit -s -m "chore: release v0.4.1"
   git push
   ```
3. Tag and push:
   ```bash
   git tag v0.4.1
   git push --tags
   ```
4. Watch Actions — both `pypi` and `npm` jobs should run
5. Verify:
   ```bash
   npm view @recall-ai/mcp version
   # 0.4.1
   ```

---

## Versioning strategy

The npm package and the Python package should stay version-aligned:

| Repo | File | Sample value |
|---|---|---|
| Python | `pyproject.toml` `version` | `"0.4.1"` |
| npm | `npm/package.json` `version` | `"0.4.1"` |
| Git | tag | `v0.4.1` |

When the npm wrapper is unchanged but Python has a patch release, bump both
anyway — keeping them aligned prevents user confusion ("which version of
Recall am I actually running?").

For pre-releases, npm uses semver dist-tags:

```bash
npm publish --tag beta              # publishes as @recall-ai/mcp@0.5.0-beta
npm install @recall-ai/mcp@beta     # users opt in
```

The `latest` dist-tag (the default that `npx -y @recall-ai/mcp` resolves)
only updates when you publish without `--tag`.

---

## Maintenance — new releases

The repeat-pattern from this point forward:

1. Make changes to Python source / npm wrapper
2. Bump versions in **both** `pyproject.toml` and `npm/package.json`
3. Update `CHANGELOG.md`
4. Commit + push
5. `git tag v0.X.Y && git push --tags`
6. CI auto-publishes to **both** PyPI and npm

Time per release: ~3 minutes of human work + ~5 minutes of CI runtime.

---

## Unpublishing & deprecating

### `npm unpublish` rules

- You can unpublish a version within **72 hours** of publishing — and only if
  it has fewer than 300 weekly downloads
- After 72 hours, you cannot unpublish without contacting npm support
- The version's slot is gone forever — you can't republish the same version

This is by design (npm learned from `left-pad` / 2016).

### `npm deprecate` (the right way)

To signal "don't use this version" without removing it:

```bash
npm deprecate @recall-ai/mcp@0.4.0 "Critical bug — please upgrade to 0.4.1"
```

`npm install @recall-ai/mcp@0.4.0` still works, but shows a yellow warning.
This is what you should do for buggy releases.

### Hard-removing a release (last resort)

If you accidentally published secrets:

1. Immediately rotate any leaked credentials (don't wait for unpublish to
   succeed)
2. Email **support@npmjs.com** with the package and version
3. They'll review and remove (typically within 24h)

---

## Troubleshooting

### `npm error 403 Forbidden — You do not have permission to publish`

Common causes:
- The unscoped name is taken — switch to scoped (`@yourorg/...`)
- You're publishing a scoped package without `--access public`
- 2FA isn't set up — npm requires it for publishing
- Token doesn't have write access to the scope/package

### `npm error E404 — `npm install` can't find the package`

Common causes:
- Typo in package name
- You published to the wrong registry (private registry config in `.npmrc`)
- npm caches; force refresh:
  ```bash
  rm -rf ~/.npm/_npx
  npm install @recall-ai/mcp --no-cache
  ```

### `recall-mcp: uvx is not installed`

The wrapper printed this. The user needs `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
brew install uv
```

If you want to support Python-without-uv, set:

```bash
RECALL_MCP_COMMAND=python RECALL_MCP_ARGS=-m,recall.mcp_server npx -y @recall-ai/mcp
```

### `recall-mcp: SyntaxError`

Your `bin/recall-mcp.js` uses Node syntax not supported by older runtimes.
Set `engines.node: ">=18"` and tell users to upgrade Node.

### Provenance attestation fails in CI

Ensure:
- `permissions: id-token: write` is set on the job
- The job runs from a public repo (provenance only works from public repos at v0.4 of the spec)
- `actions/setup-node@v4` is used (not v3)

### Publishing the same version twice

npm rejects republishing the same version. Bump the version in
`npm/package.json` and try again. (This is also why we keep PyPI and npm
versions in lockstep — easier to remember which version was published where.)

---

## Quick checklist

```
npm launch
──────────
□  Install Node 18+ and npm 9+
□  Create npmjs.com account
□  Set up 2FA, save recovery codes
□  (Optional) Create npm org @recall-ai for free
□  Decide name:  unscoped / @user-scoped / @org-scoped
□  Check name availability:  npm view @recall-ai/mcp
□  Create npm/ subdirectory in your repo
□  Write package.json with bin, files, engines, publishConfig
□  Copy LICENSE
□  Write bin/recall-mcp.js (wrapper that spawns Python via uvx)
□  Write npm/README.md
□  Test locally:  npm link  →  recall-mcp <<< '{}'
□  Test via inspector:  npx @modelcontextprotocol/inspector recall-mcp
□  npm login (with 2FA)
□  Verify package contents:  npm pack --dry-run
□  First publish:  npm publish --access public
□  Verify install:  rm -rf ~/.npm/_npx && npx -y @recall-ai/mcp
□  Configure Claude Desktop / Cursor / Codex with the npm command
□  Cross-tool demo: add_memory in Claude, search_memory in Cursor
□  Generate npm token (Granular, scoped to package)
□  Add NPM_TOKEN GitHub secret
□  Update .github/workflows/release.yml with npm publish job
□  Tag v0.4.1 and confirm CI auto-publishes
□  Document the release process (this file)
```

---

## Related

- [npm CLI documentation](https://docs.npmjs.com/cli/v10)
- [npm provenance attestations](https://docs.npmjs.com/generating-provenance-statements)
- [Model Context Protocol specification](https://modelcontextprotocol.io)
- [Recall's `release.yml` workflow](../.github/workflows/release.yml)
- [Recall's PyPI publishing guide](PUBLISH_PYPI.md)
- [Anthropic — Building MCP servers](https://modelcontextprotocol.io/quickstart/server)
