"""Graph-theoretic primitives for memory analysis.

Modules:
  spectral   — Laplacian eigenvalues, Cheeger constant, heat kernel, PageRank
  topology   — Persistent homology of memory graph (gudhi)
  transport  — Wasserstein and Gromov-Wasserstein graph-graph distance (POT)
  curvature  — Ollivier-Ricci curvature for community / pruning

These give Recall actual graph mathematics, not just SQLite + numpy.
"""

from recall.graph.curvature import (
    compute_ollivier_ricci,
    curvature_pruning_signal,
    curvature_summary,
)
from recall.graph.sheaf import (
    harmonic_dimension,
    inconsistency_score,
    sheaf_laplacian,
    signed_incidence,
)
from recall.graph.spectral import (
    cheeger_constant,
    graph_health,
    graph_laplacian,
    heat_kernel_signature,
    laplacian_eigenvalues,
    personalized_pagerank,
    spectral_gap,
)
from recall.graph.topology import (
    persistent_homology_summary,
    topological_signature,
)
from recall.graph.transport import (
    gromov_wasserstein_distance,
    wasserstein_graph_distance,
)

__all__ = [
    "graph_laplacian",
    "laplacian_eigenvalues",
    "spectral_gap",
    "cheeger_constant",
    "graph_health",
    "heat_kernel_signature",
    "personalized_pagerank",
    "persistent_homology_summary",
    "topological_signature",
    "wasserstein_graph_distance",
    "gromov_wasserstein_distance",
    "compute_ollivier_ricci",
    "curvature_pruning_signal",
    "curvature_summary",
    "sheaf_laplacian",
    "signed_incidence",
    "harmonic_dimension",
    "inconsistency_score",
]
