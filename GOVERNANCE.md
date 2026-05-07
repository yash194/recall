# Recall — Governance & Open Source Commitment

> Version 1.0 · 2026-05-07 · binding on the Recall maintainers and any future
> entity holding the repository.

## 1. The OSS-forever commitment

Recall is licensed under **Apache License 2.0** in perpetuity. The Apache-2.0
licensed codebase will always include:

1. **The full retrieval engine** — all retrieval modes (symmetric, path,
   hybrid, auto), the Γ retrieval primitive, the Personalized PageRank
   variant, the PCST/PCSF subgraph extractors.
2. **The full typed-edge runtime** — write pipeline (provenance firewall,
   quality gating, dedup, edge induction), all 7 edge types
   (supports, contradicts, corrects, agrees, pivots, temporal_next,
   superseded), the audit log.
3. **All graph mathematics** — spectral analysis (Cheeger, λ_2, heat kernel,
   PageRank), persistent homology, optimal transport, Ollivier-Ricci
   curvature, sheaf-Laplacian inconsistency detection.
4. **All hallucination bounds** — PAC-Bayes, RAG-as-noisy-ICL, spectral
   Cheeger, Conformal Risk Control.
5. **All consolidation primitives** — BMRS Bayesian pruning, mean-field
   refinement, motif extraction, PMED scoring (D_RPD, DCR, P_syco, Q_corr,
   Q_eff, Q_rare).
6. **All benchmarks** — junk replay, HotpotQA, synthetic Γ, Mem0 head-to-head,
   plus the LongMemEval and MemoryAgentBench harnesses as they're added.
7. **All integrations** — MCP server, FastAPI server, Letta adapter,
   LangGraph adapter, Mastra adapter, browser extension, personal CLI.

What may live in **Recall Cloud** (a separate paid offering) is hosting,
ops, multi-tenant isolation, audit log retention beyond 30 days, customer
support, and on-call SLAs — **never features missing from the OSS**.

## 2. Crippleware ban

The maintainers will not:

- Add a paywall to any current or future feature listed in §1.
- Move features from OSS to a paid tier under any circumstances.
- Introduce a "Business Source License" or any source-available license that
  delays Apache-2.0 reciprocity.
- Sub-license the codebase to a foundation or other entity in a way that
  weakens the Apache-2.0 commitment.

Any future entity acquiring the repository must agree to this commitment in
writing as a condition of transfer.

## 3. Decision-making

- **Day-to-day**: maintainers (`CODEOWNERS`) decide via lazy consensus.
  Disputes go to majority vote of named maintainers.
- **Architectural changes**: open RFC issues before merging changes that
  touch the public API (`Memory`, `Storage` protocol, edge type vocabulary,
  retrieval mode contract). 7-day comment window minimum.
- **Breaking changes**: only on major version bumps (semver). Each major
  version ships with a written migration guide.

## 4. Maintainer rotation

- Maintainer status is granted by majority vote of existing maintainers.
- A maintainer who is inactive for >180 days is automatically rotated to
  emeritus status; reactivation requires majority vote.
- Disagreements between maintainers about scope/direction must be resolved
  publicly in a GitHub Discussion before any code change reflects the
  resolution.

## 5. Funding & conflicts

- The maintainers may accept funding from any source (GitHub Sponsors, Open
  Collective, grants, employer time, customer contracts) provided no funder
  is granted preferential treatment in feature prioritization or roadmap.
- Any maintainer with a financial interest in a feature must disclose it in
  the PR description before merging.

## 6. Code of conduct

Contributor Covenant 2.1 applies. Reports to `conduct@recall.dev` (private)
or any maintainer.

## 7. Trademark

The "Recall" wordmark and logo (when registered) may be used freely by:
- Anyone running the OSS, including in its name and documentation
- Forks, downstreams, derivative works that retain the Apache-2.0 license

The wordmark may not be used to:
- Imply endorsement by the Recall maintainers of a third-party product
- Brand a paid hosting service in a way that confuses with `recall.dev`

## 8. Cloud tier roadmap (advisory, non-binding)

- T+30: cloud waitlist opens at recall.dev/cloud
- T+90: Recall Cloud paid tier launches (Free / Pro $29 / Team $199 / Enterprise)
- T+180: SOC 2 Type II
- T+365: governance review — this document re-ratified or amended by maintainer vote

## 9. Forking is welcome

If we ever break this commitment, the OSS-forever clause requires us to
provide a clean fork-friendly snapshot. Any maintainer leaving the project
in protest may take the codebase with them and run a parallel fork; we
will not contest that fork legally or socially.

---

*Signed by the founding maintainers as a condition of the v0.4 release.*
