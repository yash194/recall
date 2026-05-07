"""Linearize a retrieved subgraph into a context string for the LLM."""
from __future__ import annotations

from recall.types import Edge, Node


def linearize_subgraph(
    nodes: list[Node],
    edges: list[Edge],
    drawer_lookup: dict[str, str] | None = None,
    include_drawers: bool = True,
) -> str:
    """Render the subgraph as a context string for bounded_generate.

    Format:
      [N1] node1.text  (drawer: "...")
        --(edge_type, weight)--> [N2] node2.text
      [N2] ...

    Edges are listed grouped by source node.
    """
    if not nodes:
        return ""

    drawer_lookup = drawer_lookup or {}
    node_index: dict[str, int] = {n.id: i + 1 for i, n in enumerate(nodes)}
    out_lines: list[str] = []
    for n in nodes:
        idx = node_index[n.id]
        line = f"[N{idx}] ({n.role or 'fact'}) {n.text}"
        if include_drawers and n.drawer_ids:
            drawer_text = " | ".join(
                drawer_lookup[did] for did in n.drawer_ids if did in drawer_lookup
            )
            if drawer_text:
                line += f"  (verbatim: \"{drawer_text[:200]}\")"
        out_lines.append(line)
        # outgoing edges
        for e in edges:
            if e.src_node_id == n.id:
                dst_idx = node_index.get(e.dst_node_id)
                if dst_idx is None:
                    continue
                etype = e.edge_type.value if hasattr(e.edge_type, "value") else str(e.edge_type)
                out_lines.append(f"    --({etype}, w={e.weight:+.3f})--> [N{dst_idx}]")

    return "\n".join(out_lines)
