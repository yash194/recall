# typed-recall

> The npm wrapper for [Recall](https://github.com/yash194/recall) — a typed-edge memory MCP server for AI agents (Claude Code / Codex / Cursor / Windsurf).

This package is a thin Node.js shim that runs the Python Recall MCP server
via `uvx` or `pipx`. **The actual implementation is in Python.** The npm
package exists so you can install and add Recall to your MCP client with a
single command.

## Install + use

### Claude Code

```bash
claude mcp add recall -- npx -y typed-recall
```

### Cursor / Windsurf / VS Code MCP

In your client's MCP config:

```json
{
  "mcpServers": {
    "recall": {
      "command": "npx",
      "args": ["-y", "typed-recall"]
    }
  }
}
```

### Codex CLI

In `~/.codex/config.toml`:

```toml
[mcp_servers.recall]
command = "npx"
args = ["-y", "typed-recall"]
```

### Standalone

```bash
npm install -g typed-recall
typed-recall
```

## What it actually does

On first run the shim resolves a Python invocation in this order:

1. `uvx --from 'typed-recall[mcp,embed-bge,llm-openai]' typed-recall` — uses
   [uv](https://github.com/astral-sh/uv) to run the Python server in an
   ephemeral environment. **Recommended.**
2. `pipx run --spec 'typed-recall[mcp,embed-bge,llm-openai]' typed-recall` —
   [pipx](https://pipx.pypa.io) fallback.
3. `python3 -m recall.mcp_server` — assumes you already ran
   `pip install 'typed-recall[mcp,embed-bge,llm-openai]'`.

If none of these are available, it prints clear install hints.

To customize, set `RECALL_MCP_CMD` to a JSON array, e.g.
`RECALL_MCP_CMD='["python3","-m","recall.mcp_server"]'`.

To silence the launch hint, set `RECALL_MCP_QUIET=1`.

## What Recall does

A memory layer for AI agents that stores conversational memories as a
typed-edge graph, retrieves connected reasoning paths, and returns
answers structurally bounded by retrieval support:

- **Audit log** — every memory op append-only logged
- **Surgical forget** — by node-id with cascading edge deprecation
- **Bounded generation** — non-vacuous CRC hallucination bound per answer
- **Typed edges** — `supports` / `contradicts` / `superseded` / `pivots` / `temporal_next`
- **Multi-hop retrieval** — HippoRAG-style entity expansion
- **Sheaf-based inconsistency detection** — frustrated-cycle scores
- **Forman-Ricci bottleneck protection** — graph topology preserved during pruning
- **Local SQLite, zero cloud required**

Full docs: <https://github.com/yash194/recall>.

## Tools exposed via MCP

`add_memory` · `search_memory` · `bounded_answer` · `forget` · `audit` · `graph_health` · `consolidate` · `stats`

## Requirements

- Node.js 16+
- Python 3.10+
- One of: `uv` (recommended), `pipx`, or `pip install typed-recall[mcp]` already done

## License

Apache-2.0.
