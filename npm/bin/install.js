/**
 * Recall MCP installer — interactive one-time setup.
 *
 * Runs as `npx typed-recall install`.
 *
 * Steps:
 *   1. Asks which MCP client(s) to register with
 *      (Codex / Claude Code / Cursor / Windsurf / skip).
 *   2. Asks lite (~10MB, TF-IDF) vs full (~150MB, BGE + OpenAI).
 *   3. Optional OpenAI / TokenRouter key for real bounded generation.
 *   4. Installs `uv` if missing (auto, via official installer).
 *   5. Pre-warms the uvx cache so subsequent MCP starts are instant.
 *   6. Writes the MCP entry to each chosen client's config.
 *   7. Prints next steps (restart the client to pick it up).
 *
 * Non-interactive flags (skip prompts):
 *   --client <codex|claude-code|cursor|windsurf|all|none>
 *   --version <lite|full>
 *   --openai-key <KEY>
 *   --openai-base-url <URL>
 *   --openai-model <MODEL>
 *   --yes  (skip every prompt; use defaults: codex+claude-code, lite, no key)
 */
import { spawn, spawnSync } from "node:child_process";
import { existsSync, readFileSync, writeFileSync, mkdirSync, appendFileSync } from "node:fs";
import { homedir, platform } from "node:os";
import { join, dirname } from "node:path";
import readline from "node:readline/promises";
import { stdin, stdout } from "node:process";

const HOME = homedir();
const IS_WIN = platform() === "win32";

// ----------------- helpers -----------------

function which(cmd) {
  const r = spawnSync(IS_WIN ? "where" : "which", [cmd], { encoding: "utf8" });
  return r.status === 0 ? (r.stdout || "").trim().split(/\r?\n/)[0] : null;
}

function findUvx() {
  const candidates = [
    which("uvx"),
    join(HOME, ".local", "bin", IS_WIN ? "uvx.exe" : "uvx"),
    join(HOME, ".cargo", "bin", IS_WIN ? "uvx.exe" : "uvx"),
    "/usr/local/bin/uvx",
    "/opt/homebrew/bin/uvx",
  ].filter(Boolean);
  for (const p of candidates) if (existsSync(p)) return p;
  return null;
}

function parseFlags(argv) {
  const flags = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--yes" || a === "-y") flags.yes = true;
    else if (a.startsWith("--")) flags[a.slice(2).replace(/-/g, "_")] = argv[++i];
  }
  return flags;
}

function detectClients() {
  return {
    codex: existsSync(join(HOME, ".codex", "config.toml"))
      || existsSync(join(HOME, ".codex")),
    claude_code: !!which("claude"),
    cursor: existsSync(join(HOME, ".cursor"))
      || existsSync(join(HOME, "Library", "Application Support", "Cursor")),
    windsurf: existsSync(join(HOME, ".codeium", "windsurf"))
      || existsSync(join(HOME, "Library", "Application Support", "Windsurf")),
  };
}

// ----------------- prompts -----------------

async function prompt(rl, label, defaultValue) {
  const suffix = defaultValue ? ` [${defaultValue}]` : "";
  const ans = (await rl.question(`${label}${suffix}: `)).trim();
  return ans || defaultValue || "";
}

async function promptYesNo(rl, label, defaultYes = false) {
  const def = defaultYes ? "Y/n" : "y/N";
  const ans = (await rl.question(`${label} [${def}]: `)).trim().toLowerCase();
  if (!ans) return defaultYes;
  return /^y/.test(ans);
}

async function promptChoice(rl, label, choices, defaultChoice) {
  const list = choices.map((c, i) => `  ${i + 1}. ${c.label}${c.detail ? "  — " + c.detail : ""}`).join("\n");
  const ans = (await rl.question(`${label}\n${list}\nChoice [${defaultChoice}]: `)).trim();
  if (!ans) return defaultChoice;
  const num = parseInt(ans, 10);
  if (Number.isInteger(num) && num >= 1 && num <= choices.length) return choices[num - 1].value;
  // also allow value strings ("codex", "all", etc.)
  const match = choices.find(c => c.value === ans.toLowerCase() || c.label.toLowerCase().startsWith(ans.toLowerCase()));
  if (match) return match.value;
  return defaultChoice;
}

// ----------------- install steps -----------------

function installUv() {
  const existing = findUvx();
  if (existing) {
    console.log(`✓ uv already installed at ${existing}`);
    return existing;
  }
  if (IS_WIN) {
    console.error("uv auto-install isn't supported on Windows in this script.");
    console.error("Install uv manually: https://docs.astral.sh/uv/getting-started/installation/");
    return null;
  }
  console.log("Installing uv (~30MB, one-time)...");
  const r = spawnSync("sh", ["-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"], {
    stdio: ["ignore", "inherit", "inherit"],
  });
  if (r.status !== 0) {
    console.error("uv installer failed.");
    return null;
  }
  return findUvx();
}

function preWarmCache(uvx, extras) {
  console.log(`Pre-fetching Python MCP server (extras: [${extras}])...`);
  console.log(`  This downloads the Python deps into uv's cache so MCP starts are instant.`);
  console.log(`  (One-time. Lite ≈ 10MB; Full ≈ 150MB with torch + transformers.)\n`);
  // Run `uvx --from typed-recall[<extras>] recall-mcp` and feed it an empty
  // stdin so the MCP server immediately exits after handshake init.
  const child = spawn(
    uvx,
    ["--from", `typed-recall[${extras}]`, "recall-mcp"],
    { stdio: ["pipe", "inherit", "inherit"] },
  );
  child.stdin.end();
  return new Promise(resolve => child.on("exit", () => resolve()));
}

// ----------------- config writers -----------------

function makeMcpEnvBlock(opts) {
  const out = {};
  if (opts.extras && opts.extras !== "mcp") out.RECALL_MCP_EXTRAS = opts.extras;
  if (opts.openai_key) out.OPENAI_API_KEY = opts.openai_key;
  if (opts.openai_base_url) out.OPENAI_BASE_URL = opts.openai_base_url;
  if (opts.openai_model) out.RECALL_OPENAI_MODEL = opts.openai_model;
  if (opts.db_dir) out.RECALL_DB_DIR = opts.db_dir;
  return out;
}

function writeCodexConfig(opts) {
  const path = join(HOME, ".codex", "config.toml");
  mkdirSync(dirname(path), { recursive: true });
  let content = existsSync(path) ? readFileSync(path, "utf8") : "";
  // Strip any prior [mcp_servers.recall] block
  content = content.replace(
    /\n*# ─── Recall.*?\n\[mcp_servers\.recall\][\s\S]*?(?=\n\[(?!mcp_servers\.recall)|\n*$)/g,
    "",
  );
  content = content.replace(
    /\n*\[mcp_servers\.recall\][\s\S]*?(?=\n\[(?!mcp_servers\.recall\.)|\n*$)/g,
    "",
  );
  content = content.replace(
    /\n*\[mcp_servers\.recall\.env\][\s\S]*?(?=\n\[(?!mcp_servers\.recall\.)|\n*$)/g,
    "",
  );
  const env = makeMcpEnvBlock(opts);
  let block = "\n# ─── Recall — typed-edge memory MCP server ───\n";
  block += "[mcp_servers.recall]\n";
  block += `command = "npx"\n`;
  block += `args = ["-y", "typed-recall@latest"]\n`;
  if (Object.keys(env).length > 0) {
    block += "[mcp_servers.recall.env]\n";
    for (const [k, v] of Object.entries(env)) {
      block += `${k} = "${v.replace(/"/g, '\\"')}"\n`;
    }
  }
  writeFileSync(path, content.trimEnd() + "\n" + block);
  console.log(`✓ Wrote MCP entry to ${path}`);
}

function writeClaudeCodeConfig(opts) {
  if (!which("claude")) {
    console.log("⚠ Claude Code CLI (`claude`) not found on PATH. Skipping.");
    console.log("  To register manually: claude mcp add recall -- npx -y typed-recall");
    return false;
  }
  // Remove any prior recall registration (best-effort)
  spawnSync("claude", ["mcp", "remove", "recall", "--scope", "user"], { stdio: "ignore" });
  // Add fresh
  const args = ["mcp", "add", "recall", "--scope", "user"];
  const env = makeMcpEnvBlock(opts);
  for (const [k, v] of Object.entries(env)) {
    args.push("-e", `${k}=${v}`);
  }
  args.push("--", "npx", "-y", "typed-recall@latest");
  const r = spawnSync("claude", args, { stdio: "inherit" });
  if (r.status === 0) {
    console.log("✓ Registered with Claude Code (user scope)");
    return true;
  }
  console.log("⚠ `claude mcp add` failed. You can run it manually:");
  console.log("    claude mcp add recall -- npx -y typed-recall@latest");
  return false;
}

function writeCursorConfig(opts) {
  const path = join(HOME, ".cursor", "mcp.json");
  mkdirSync(dirname(path), { recursive: true });
  let cfg = { mcpServers: {} };
  if (existsSync(path)) {
    try { cfg = JSON.parse(readFileSync(path, "utf8")); } catch { /* keep default */ }
    if (!cfg.mcpServers) cfg.mcpServers = {};
  }
  const env = makeMcpEnvBlock(opts);
  cfg.mcpServers.recall = {
    command: "npx",
    args: ["-y", "typed-recall@latest"],
    ...(Object.keys(env).length > 0 ? { env } : {}),
  };
  writeFileSync(path, JSON.stringify(cfg, null, 2));
  console.log(`✓ Wrote MCP entry to ${path}`);
}

function writeWindsurfConfig(opts) {
  const path = join(HOME, ".codeium", "windsurf", "mcp_config.json");
  mkdirSync(dirname(path), { recursive: true });
  let cfg = { mcpServers: {} };
  if (existsSync(path)) {
    try { cfg = JSON.parse(readFileSync(path, "utf8")); } catch { /* keep default */ }
    if (!cfg.mcpServers) cfg.mcpServers = {};
  }
  const env = makeMcpEnvBlock(opts);
  cfg.mcpServers.recall = {
    command: "npx",
    args: ["-y", "typed-recall@latest"],
    ...(Object.keys(env).length > 0 ? { env } : {}),
  };
  writeFileSync(path, JSON.stringify(cfg, null, 2));
  console.log(`✓ Wrote MCP entry to ${path}`);
}

// ----------------- main -----------------

export async function runInstaller(argv = []) {
  const flags = parseFlags(argv);
  const detected = detectClients();

  console.log("\nRecall MCP installer\n");
  console.log("This will:");
  console.log("  1. Install `uv` (Python tool runner) if not present");
  console.log("  2. Pre-fetch the Python MCP server into uv's cache");
  console.log("  3. Register the MCP entry with your chosen client(s)\n");

  let chosenClients = [];
  let extras = "mcp";
  let openaiKey = "";
  let openaiBaseUrl = "";
  let openaiModel = "";

  if (flags.yes || flags.client) {
    // non-interactive
    const c = (flags.client || "codex,claude-code").toLowerCase();
    if (c === "all") chosenClients = ["codex", "claude_code", "cursor", "windsurf"];
    else if (c === "none") chosenClients = [];
    else chosenClients = c.split(",").map(s => s.trim().replace(/-/g, "_")).filter(Boolean);
    extras = flags.version === "full" ? "mcp,embed-bge,llm-openai" : "mcp";
    openaiKey = flags.openai_key || "";
    openaiBaseUrl = flags.openai_base_url || "";
    openaiModel = flags.openai_model || "";
  } else {
    const rl = readline.createInterface({ input: stdin, output: stdout });

    // Show detected clients
    console.log("Detected MCP clients on this machine:");
    for (const [k, v] of Object.entries(detected)) {
      console.log(`  ${v ? "✓" : "·"} ${k.replace("_", "-")}`);
    }
    console.log();

    const clientChoice = await promptChoice(rl, "Which client(s) to register with?", [
      { value: "codex",        label: "Codex CLI",        detail: "writes ~/.codex/config.toml" },
      { value: "claude_code",  label: "Claude Code",      detail: "calls `claude mcp add recall` (user scope)" },
      { value: "cursor",       label: "Cursor",           detail: "writes ~/.cursor/mcp.json" },
      { value: "windsurf",     label: "Windsurf",         detail: "writes ~/.codeium/windsurf/mcp_config.json" },
      { value: "all",          label: "All of the above", detail: "" },
      { value: "skip",         label: "Skip — just install uv + cache" },
    ], "all");

    if (clientChoice === "all") chosenClients = ["codex", "claude_code", "cursor", "windsurf"];
    else if (clientChoice === "skip") chosenClients = [];
    else chosenClients = [clientChoice];

    const wantFull = await promptYesNo(rl, "Install full version (BGE neural embeddings + OpenAI client, ~150MB)?", false);
    extras = wantFull ? "mcp,embed-bge,llm-openai" : "mcp";

    if (extras.includes("llm-openai")) {
      const wantKey = await promptYesNo(rl, "Configure an OpenAI / TokenRouter API key now?", false);
      if (wantKey) {
        openaiKey = await prompt(rl, "  API key (input echoed — paste in a private window if needed)");
        openaiBaseUrl = await prompt(rl, "  Base URL", "https://api.openai.com/v1");
        openaiModel = await prompt(rl, "  Model name", "gpt-4o-mini");
      }
    }

    rl.close();
    console.log();
  }

  // Step 1: install uv if needed
  const uvx = installUv();
  if (!uvx) {
    console.error("\n❌ Could not install uv. Aborting.");
    process.exit(1);
  }

  // Step 2: pre-warm cache
  await preWarmCache(uvx, extras);

  // Step 3: write configs
  console.log("\nRegistering MCP entry:");
  const opts = {
    extras,
    openai_key: openaiKey,
    openai_base_url: openaiBaseUrl,
    openai_model: openaiModel,
  };
  for (const c of chosenClients) {
    if (c === "codex") writeCodexConfig(opts);
    else if (c === "claude_code") writeClaudeCodeConfig(opts);
    else if (c === "cursor") writeCursorConfig(opts);
    else if (c === "windsurf") writeWindsurfConfig(opts);
  }

  console.log("\n✅ Done.\n");
  if (chosenClients.length > 0) {
    console.log("Restart the client(s) above to pick up the MCP server. The first MCP-tool call will be instant — uv's cache is already warmed.\n");
  } else {
    console.log("Cache is warmed. To register manually, see https://github.com/yash194/recall#installation\n");
  }
  console.log("Tools exposed: add_memory · search_memory · bounded_answer · forget · audit · graph_health · consolidate · stats\n");
}

// CLI entry — can be invoked directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runInstaller(process.argv.slice(2)).catch(err => {
    console.error(err.stack || err);
    process.exit(1);
  });
}
