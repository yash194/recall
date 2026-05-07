#!/usr/bin/env node
/**
 * recall-mcp — npm shim that runs the Python Recall MCP server.
 *
 * Strategy:
 *   1. Try `uvx --from typed-recall recall-mcp` (uv-managed, no global install needed).
 *   2. If uvx is not available, fall back to `pipx run --spec typed-recall recall-mcp`.
 *   3. If neither, fall back to `python3 -m recall.mcp_server` (assumes
 *      `pip install typed-recall[mcp]` was already done).
 *
 * The shim is a transparent stdio passthrough — it forwards stdin/stdout/stderr
 * and propagates the exit code so MCP clients (Claude Code, Codex, Cursor,
 * Windsurf) can talk to the server unmodified.
 *
 * To customize the underlying invocation, set RECALL_MCP_CMD to a JSON-array
 * string (e.g. RECALL_MCP_CMD='["python3","-m","recall.mcp_server"]').
 */
import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";

const env = { ...process.env };

// Resolve which underlying command to use.
function which(cmd) {
  const r = spawnSync(process.platform === "win32" ? "where" : "which", [cmd], {
    encoding: "utf8",
  });
  if (r.status !== 0) return null;
  const out = (r.stdout || "").trim().split(/\r?\n/)[0];
  return out && existsSync(out) ? out : null;
}

function resolveCommand() {
  // Honor RECALL_MCP_CMD override — JSON array of [executable, ...args].
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
  if (which("uvx")) {
    return {
      cmd: "uvx",
      args: ["--from", "typed-recall[mcp,embed-bge,llm-openai]", "recall-mcp"],
      via: "uvx",
    };
  }
  if (which("pipx")) {
    return {
      cmd: "pipx",
      args: ["run", "--spec", "typed-recall[mcp,embed-bge,llm-openai]", "recall-mcp"],
      via: "pipx",
    };
  }
  // Last resort: assume `pip install recall[mcp]` was done already.
  const py =
    which("python3") || which("python") || (process.platform === "win32" ? "python" : "python3");
  return { cmd: py, args: ["-m", "recall.mcp_server"], via: "python3" };
}

const { cmd, args, via } = resolveCommand();

// Pass user args through (rare for MCP servers, but safe).
const userArgs = process.argv.slice(2);
const allArgs = [...args, ...userArgs];

// Surface a stderr hint on first run so users know what's happening.
if (!env.RECALL_MCP_QUIET) {
  process.stderr.write(`[recall-mcp] launching via ${via}: ${cmd} ${allArgs.join(" ")}\n`);
}

const child = spawn(cmd, allArgs, {
  stdio: "inherit",
  env,
});

child.on("error", (err) => {
  if (err.code === "ENOENT") {
    process.stderr.write(
      `[recall-mcp] ERROR: \`${cmd}\` not found. Install one of:\n` +
        `  - uv (recommended):  curl -LsSf https://astral.sh/uv/install.sh | sh\n` +
        `  - pipx:              python3 -m pip install --user pipx && pipx ensurepath\n` +
        `  - or: pip install 'typed-recall[mcp,embed-bge,llm-openai]'\n`,
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

// Forward signals to the child so Ctrl-C / SIGTERM work cleanly.
for (const sig of ["SIGINT", "SIGTERM", "SIGHUP"]) {
  process.on(sig, () => {
    if (!child.killed) child.kill(sig);
  });
}
