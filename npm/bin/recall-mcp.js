#!/usr/bin/env node
/**
 * recall-mcp — npm shim for Recall.
 *
 * Two modes:
 *
 *   $ npx typed-recall install              ← INTERACTIVE setup wizard
 *                                              (asks which client, lite/full,
 *                                              optional API key, then writes
 *                                              the MCP entry to your config
 *                                              and pre-warms uv's cache so
 *                                              MCP starts are instant)
 *
 *   $ npx -y typed-recall                   ← MCP SERVER mode (default)
 *                                              spawned by Codex/Claude Code/
 *                                              Cursor/Windsurf as the MCP
 *                                              process; reads JSON-RPC on
 *                                              stdin, writes responses on
 *                                              stdout, transparent passthrough
 *                                              to the Python server.
 *
 * v0.2.0 — adds `install` subcommand for one-time, opt-in setup so
 *          MCP-client startup never has to download Python deps live.
 *
 * The default extras are now just `[mcp]` so first-run install is small
 * (~10MB vs ~150MB with torch/transformers). TF-IDF embedder is the
 * default and works without any heavy ML deps. Set
 * `RECALL_MCP_EXTRAS=mcp,embed-bge,llm-openai` if you want neural BGE
 * embeddings + OpenAI client compiled in.
 *
 * Resolution order:
 *   1. RECALL_MCP_CMD env override (JSON-array string).
 *   2. `uvx --from 'typed-recall[<extras>]' recall-mcp` (uv-managed).
 *   3. `pipx run --spec 'typed-recall[<extras>]' recall-mcp` (pipx fallback).
 *   4. If neither uvx nor pipx is on PATH, auto-install `uv` via the
 *      official Astral installer, then retry with uvx.
 *   5. Last resort: `python3 -m recall.mcp_server` (assumes the user
 *      already ran `pip install 'typed-recall[mcp]'`).
 *
 * The shim is a transparent stdio passthrough — it forwards
 * stdin/stdout/stderr and propagates the exit code so MCP clients
 * (Claude Code, Codex, Cursor, Windsurf) can talk to the server unmodified.
 */
import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

// ---- subcommand dispatch ----
// `npx typed-recall install [...]`  → run the interactive installer
// `npx typed-recall help|--help|-h` → print usage
// anything else (or no args)         → MCP server mode

const sub = process.argv[2];

if (sub === "install" || sub === "init" || sub === "setup") {
  const here = dirname(fileURLToPath(import.meta.url));
  const { runInstaller } = await import(join(here, "install.js"));
  await runInstaller(process.argv.slice(3));
  process.exit(0);
}

if (sub === "help" || sub === "--help" || sub === "-h") {
  process.stdout.write(
    `\nRecall MCP — typed-edge memory layer for AI agents.\n\n` +
      `Usage:\n` +
      `  npx typed-recall install      Interactive one-time setup wizard\n` +
      `  npx -y typed-recall           Start the MCP server (used by clients)\n` +
      `  npx typed-recall help         Show this help\n\n` +
      `For the install command:\n` +
      `  --yes                  Use defaults (skip prompts)\n` +
      `  --client <id>          codex | claude-code | cursor | windsurf | all | none\n` +
      `  --version <id>         lite | full\n` +
      `  --openai-key <KEY>\n` +
      `  --openai-base-url <URL>\n` +
      `  --openai-model <NAME>\n\n` +
      `Docs: https://github.com/yash194/recall\n\n`,
  );
  process.exit(0);
}

const env = { ...process.env };

// Lightweight default — pulls only `mcp>=1.0` plus the core recall lib.
// First-run install is ~10MB (vs ~150MB with embed-bge). Users who want
// BGE neural embeddings or the OpenAI client baked in can override:
//   RECALL_MCP_EXTRAS=mcp,embed-bge,llm-openai
const EXTRAS = env.RECALL_MCP_EXTRAS || "mcp";
const PKG_SPEC = `typed-recall[${EXTRAS}]`;

function which(cmd) {
  const r = spawnSync(process.platform === "win32" ? "where" : "which", [cmd], {
    encoding: "utf8",
  });
  if (r.status !== 0) return null;
  const out = (r.stdout || "").trim().split(/\r?\n/)[0];
  return out && existsSync(out) ? out : null;
}

// uv installs itself to ~/.local/bin (Linux/macOS) or ~/.cargo/bin in
// some setups. Check both even before they're on PATH.
function findUvxAfterInstall() {
  const home = homedir();
  const candidates = [
    process.platform === "win32" ? join(home, ".local", "bin", "uvx.exe") : null,
    join(home, ".local", "bin", "uvx"),
    join(home, ".cargo", "bin", "uvx"),
    "/usr/local/bin/uvx",
    "/opt/homebrew/bin/uvx",
  ].filter(Boolean);
  for (const path of candidates) {
    if (existsSync(path)) return path;
  }
  return which("uvx");
}

// Auto-install uv via the official Astral installer.
// Only runs on macOS / Linux (Windows users get a clear hint).
function autoInstallUv() {
  if (process.platform === "win32") {
    process.stderr.write(
      `[recall-mcp] uv auto-install isn't supported on Windows yet.\n` +
        `[recall-mcp] please install uv manually: https://docs.astral.sh/uv/getting-started/installation/\n`,
    );
    return null;
  }
  if (env.RECALL_MCP_NO_AUTO_INSTALL) {
    process.stderr.write(
      `[recall-mcp] RECALL_MCP_NO_AUTO_INSTALL is set; skipping uv install.\n`,
    );
    return null;
  }
  process.stderr.write(
    `[recall-mcp] uv not found — installing via the official installer (one-time, ~30MB)...\n` +
      `[recall-mcp] (set RECALL_MCP_NO_AUTO_INSTALL=1 to disable this behavior)\n`,
  );
  const installer = spawnSync(
    "sh",
    ["-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"],
    {
      stdio: ["ignore", "inherit", "inherit"],
      env,
    },
  );
  if (installer.status !== 0) {
    process.stderr.write(
      `[recall-mcp] uv installer exited with status ${installer.status}.\n` +
        `[recall-mcp] Try installing manually: curl -LsSf https://astral.sh/uv/install.sh | sh\n`,
    );
    return null;
  }
  const uvxPath = findUvxAfterInstall();
  if (uvxPath) {
    process.stderr.write(`[recall-mcp] uv installed at ${uvxPath}\n`);
  }
  return uvxPath;
}

function resolveCommand() {
  if (env.RECALL_MCP_CMD) {
    try {
      const parsed = JSON.parse(env.RECALL_MCP_CMD);
      if (Array.isArray(parsed) && parsed.length >= 1) {
        return { cmd: parsed[0], args: parsed.slice(1), via: "RECALL_MCP_CMD" };
      }
    } catch (e) {
      // fall through
    }
  }
  let uvx = which("uvx") || findUvxAfterInstall();
  if (uvx) {
    return {
      cmd: uvx,
      args: ["--from", PKG_SPEC, "recall-mcp"],
      via: "uvx",
    };
  }
  if (which("pipx")) {
    return {
      cmd: "pipx",
      args: ["run", "--spec", PKG_SPEC, "recall-mcp"],
      via: "pipx",
    };
  }
  // Neither uvx nor pipx on PATH — auto-install uv.
  uvx = autoInstallUv();
  if (uvx) {
    // Make sure ~/.local/bin is in PATH for any subprocesses uvx spawns.
    const localBin = join(homedir(), ".local", "bin");
    env.PATH = `${localBin}:${env.PATH || ""}`;
    return {
      cmd: uvx,
      args: ["--from", PKG_SPEC, "recall-mcp"],
      via: "uvx (auto-installed)",
    };
  }
  // Last resort: assume `pip install typed-recall[mcp]` was done already.
  const py =
    which("python3") || which("python") || (process.platform === "win32" ? "python" : "python3");
  return { cmd: py, args: ["-m", "recall.mcp_server"], via: "python3" };
}

const { cmd, args, via } = resolveCommand();
const userArgs = process.argv.slice(2);
const allArgs = [...args, ...userArgs];

if (!env.RECALL_MCP_QUIET) {
  process.stderr.write(`[recall-mcp] launching via ${via}: ${cmd} ${allArgs.join(" ")}\n`);
}

const child = spawn(cmd, allArgs, { stdio: "inherit", env });

child.on("error", (err) => {
  if (err.code === "ENOENT") {
    process.stderr.write(
      `[recall-mcp] ERROR: \`${cmd}\` not found. Install one of:\n` +
        `  - uv (recommended):  curl -LsSf https://astral.sh/uv/install.sh | sh\n` +
        `  - pipx:              python3 -m pip install --user pipx && pipx ensurepath\n` +
        `  - or: pip install '${PKG_SPEC}'\n`,
    );
  } else {
    process.stderr.write(`[recall-mcp] spawn error: ${err.message}\n`);
  }
  process.exit(1);
});

child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
  } else {
    process.exit(code ?? 0);
  }
});

for (const sig of ["SIGINT", "SIGTERM", "SIGHUP"]) {
  process.on(sig, () => {
    if (!child.killed) child.kill(sig);
  });
}
