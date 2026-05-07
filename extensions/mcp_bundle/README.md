# Recall MCP Server

Model Context Protocol server for Recall. Drops Recall's typed-edge memory
into Claude Desktop, Cursor, Cline, Continue, Zed, Windsurf, and any other
MCP-aware tool.

## Install

### Claude Desktop / Claude Code

```bash
claude mcp add recall -- uvx --from recall recall-mcp
```

Or edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "recall": {
      "command": "uvx",
      "args": ["--from", "recall", "recall-mcp"]
    }
  }
}
```

### Cursor

Settings → MCP → Add server → command: `uvx --from recall recall-mcp`

### Cline / Continue / Zed / Windsurf

Same pattern; consult each tool's MCP settings.

## Tools exposed

| Tool | What it does |
|---|---|
| `add_memory` | Store a memory with junk-gating, dedup, and Γ-edge induction |
| `search_memory` | Return connected reasoning subgraph for a query |
| `bounded_answer` | Generate an answer bounded by structural support |
| `forget` | Surgical delete of a memory (audit-logged) |
| `audit` | Provenance trail for any operation |
| `graph_health` | Spectral / topology / curvature health metrics |
| `consolidate` | Run sleep-time consolidation (BMRS + mean-field + motif) |
| `stats` | Active nodes / edges / audit count |

## Verification

```
$ uvx --from recall recall-mcp
{"jsonrpc": "2.0", ...}
```

Speaks MCP over stdio. The Anthropic-recommended way to verify:

```bash
npx @modelcontextprotocol/inspector uvx --from recall recall-mcp
```

## Storage

By default, per-tenant SQLite at `~/.recall/<tenant>.db`. Override via
`RECALL_DB_DIR` env var.

Each MCP client typically uses the default `user` tenant; pass `tenant=...`
to any tool to scope memory to a project / team.
