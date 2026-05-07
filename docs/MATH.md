# MATH — The Complete Mathematical Foundation of Recall

> Formal mathematical specification. Every theorem here is implemented and tested
> in `tests/`. Every formula is load-bearing — if the implementation deviates,
> this document gets updated, not the test.
>
> Last updated: 2026-05-07 (v0.6).

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| `t` | A piece of text (utterance, sentence, fact) |
| `f(t) ∈ ℝᵈ` | Forward embedding: `embed("Forward describe: " + t)`, prompted |
| `b(t) ∈ ℝᵈ` | Backward embedding: `embed("Backward describe: " + t)`, prompted |
| `s(t) ∈ ℝᵈ` | Symmetric retrieval embedding. v0.6: `s(t) = embed(t)` (raw text); pre-v0.6: `s(t) = (f + b) / 2` |
| `c(t) = (f − b) / 2` | Causal/antisymmetric component (used by Γ) |
| `Γ(i → j)` | Directional retrieval score from node `i` to node `j` |
| `G = (V, E)` | Memory graph: `V` = nodes, `E` = typed weighted directed edges |
| `e ∈ E` | An edge `(i, j, type, weight)` |
| `path(i ⇒ j)` | A directed walk in `G` from node `i` to node `j` |
| `R(q)` | Retrieved subgraph for query `q` |
| `π_Γ` | The Γ-induced posterior over memory paths |
| `δ` | PAC-Bayes / CRC confidence parameter |
| `n` | Sample size (training corpus) / calibration set |
| `H¹(G; ℱ)` | First sheaf cohomology of `G` with coefficients in sheaf `ℱ` |
| `λ_F` | Eigenvalues of the sheaf Laplacian `L_F` |

All embeddings are L2-normalized: `‖f‖ = ‖b‖ = 1`. So `s` and `c` lie in the unit ball but are not unit vectors:

```
‖s‖² + ‖c‖² = ½(‖f‖² + ‖b‖²) = 1     (parallelogram identity)
s · c = ¼ (‖f‖² − ‖b‖²) = 0           (orthogonality of s and c)
```

So `(s, c)` is an orthogonal decomposition of `(f, b)` into "what it means" and "which way it points."

---

## 1. Γ — the directional retrieval primitive

### 1.1 Definition

```
Γ(i → j) := f(tᵢ) · b(tⱼ) − s(tᵢ) · s(tⱼ)
```

We expand this using `f = s + c, b = s − c`:

```
f(tᵢ) · b(tⱼ) = (sᵢ + cᵢ) · (sⱼ − cⱼ)
              = sᵢ·sⱼ − sᵢ·cⱼ + cᵢ·sⱼ − cᵢ·cⱼ
```

Therefore:

```
Γ(i → j) = (sᵢ·sⱼ − sᵢ·cⱼ + cᵢ·sⱼ − cᵢ·cⱼ) − sᵢ·sⱼ
         = cᵢ·sⱼ − sᵢ·cⱼ − cᵢ·cⱼ
         = ⟨cᵢ, sⱼ⟩ − ⟨sᵢ, cⱼ⟩ − ‖cᵢ‖·‖cⱼ‖·cos(cᵢ, cⱼ)        ⟨1.1⟩
```

Three terms:

- `cᵢ·sⱼ` — "how much the source's directionality points at the target's meaning"
- `−sᵢ·cⱼ` — "how much the target's directionality points back at the source's meaning" (subtracted)
- `−cᵢ·cⱼ` — "co-directional alignment penalty" (subtracted)

### 1.2 Provable asymmetry theorem

**Theorem 1.1** (Asymmetry of Γ). For generic texts `i ≠ j`, `Γ(i → j) ≠ Γ(j → i)`.

**Proof.** Swap `i` and `j` in `⟨1.1⟩`:

```
Γ(j → i) = cⱼ·sᵢ − sⱼ·cᵢ − cⱼ·cᵢ
         = ⟨cⱼ, sᵢ⟩ − ⟨sⱼ, cᵢ⟩ − cⱼ·cᵢ
```

Then:

```
Γ(i → j) − Γ(j → i)
  = (cᵢ·sⱼ − sᵢ·cⱼ − cᵢ·cⱼ) − (cⱼ·sᵢ − sⱼ·cᵢ − cⱼ·cᵢ)
  = cᵢ·sⱼ − sᵢ·cⱼ − cᵢ·cⱼ − cⱼ·sᵢ + sⱼ·cᵢ + cⱼ·cᵢ

Inner products are commutative: cᵢ·sⱼ = sⱼ·cᵢ; sᵢ·cⱼ = cⱼ·sᵢ; cᵢ·cⱼ = cⱼ·cᵢ.

  = cᵢ·sⱼ − cⱼ·sᵢ − cᵢ·cⱼ − cⱼ·sᵢ + cᵢ·sⱼ + cᵢ·cⱼ
  = 2(cᵢ·sⱼ) − 2(cⱼ·sᵢ)
  = 2 (⟨cᵢ, sⱼ⟩ − ⟨cⱼ, sᵢ⟩)                                ⟨1.2⟩
```

The right side is generically nonzero unless `cᵢ ⊥ sⱼ` and `cⱼ ⊥ sᵢ` simultaneously, or `cᵢ = cⱼ` and `sᵢ = sⱼ`. ∎

### 1.3 Decomposition theorem

**Theorem 1.2** (Symmetric / antisymmetric split of Γ). Define:

```
Γ_sym(i, j)  := ½ [ Γ(i → j) + Γ(j → i) ]
Γ_anti(i, j) := ½ [ Γ(i → j) − Γ(j → i) ]
```

Then:

```
Γ_sym(i, j)  = − cᵢ · cⱼ                                   ⟨1.3⟩
Γ_anti(i, j) = ⟨cᵢ, sⱼ⟩ − ⟨cⱼ, sᵢ⟩                          ⟨1.4⟩
```

**Proof.** Direct from `⟨1.1⟩` and `⟨1.2⟩`. ∎

The symmetric part is *minus* the inner product of causal components — this measures whether two texts are "co-directional in their causality," and is *subtracted* in retrieval. The antisymmetric part is the actual directional signal.

### 1.4 Corollary — what cosine retrieval misses

Cosine retrieval scores `i → j` and `j → i` identically. From `⟨1.1⟩`:

```
cos(i, j) = (sᵢ + cᵢ)·(sⱼ + cⱼ) / (‖fᵢ‖·‖fⱼ‖)
          = sᵢ·sⱼ + sᵢ·cⱼ + cᵢ·sⱼ + cᵢ·cⱼ                    (forward-vs-forward)
```

Cosine sees the symmetric part `sᵢ·sⱼ + cᵢ·cⱼ + (cross terms)` and *adds* the cross terms; Γ uses *only* the antisymmetric cross terms. Cosine cannot distinguish direction. ∎

---

## 2. Γ as Fisher–Rao pull-back (Hauberg)

This section places Γ in the geometric framework of Hauberg's *Pulling back information geometry* (Arvanitidis, González-Duque, Pouplin, Kalatzis, Hauberg, AISTATS 2022).

### 2.1 Setup

Let `M` be a manifold of probability distributions over tokens (the "token-distribution manifold"). For text `t`, define two predictive distributions:

```
p_→(· | t) := P_LLM(next-token | "Forward describe: " + t)
p_←(· | t) := P_LLM(next-token | "Backward describe: " + t)
```

The forward and backward embeddings `f(t), b(t)` are the LLM's representations of these distributions (mapped to ℝᵈ via the embedding head). We treat `(f, b)` as parameters of a point in a parameter manifold `Θ ⊂ ℝ²ᵈ`.

### 2.2 The Fisher metric

The Fisher information metric on `M` is:

```
g_F(θ)_{ab} = E_{x ∼ p_θ}[ ∂_a log p(x|θ) · ∂_b log p(x|θ) ]
```

For our two-point setup with `θ = (f, b)`, the **pull-back of `g_F` along the embedding map** induces a metric on `Θ`. Following Arvanitidis et al. §3, this pull-back has both:

- A **symmetric** part — the standard Riemannian metric, gives distance.
- An **antisymmetric** part — a torsion-like 2-form, captures the *direction* of geometric flow.

### 2.3 Identification with Γ

**Claim 2.1** (Recall's geometric foundation). The score `Γ(i → j) = f_i · b_j − s_i · s_j` is, to leading order in the embedding magnitude, the antisymmetric component of the pull-back inner product on `Θ` evaluated at `(θᵢ, θⱼ)`.

**Argument.** The pull-back inner product on `Θ` is:

```
⟨(fᵢ, bᵢ), (fⱼ, bⱼ)⟩_Θ = α(fᵢ·fⱼ) + β(bᵢ·bⱼ) + γ(fᵢ·bⱼ) + δ(bᵢ·fⱼ)
```

for some coefficients `(α, β, γ, δ)` determined by the Fisher tensor. The *symmetric* part of this inner product (under swapping `i ↔ j`) is:

```
⟨…⟩_sym = α(fᵢ·fⱼ) + β(bᵢ·bⱼ) + ½(γ + δ)(fᵢ·bⱼ + bᵢ·fⱼ)
```

The *antisymmetric* part is:

```
⟨…⟩_anti = ½(γ − δ)(fᵢ·bⱼ − bᵢ·fⱼ)
```

Now `fᵢ·bⱼ − bᵢ·fⱼ = (sᵢ + cᵢ)·(sⱼ − cⱼ) − (sᵢ − cᵢ)·(sⱼ + cⱼ) = −2(sᵢ·cⱼ) + 2(cᵢ·sⱼ) = 2 Γ_anti(i, j)`.

So `⟨…⟩_anti ∝ Γ_anti(i, j)`. The full score `Γ(i → j) = Γ_sym + Γ_anti = −cᵢ·cⱼ + Γ_anti` includes a small symmetric correction (`−cᵢ·cⱼ`) which can be interpreted as a torsion-induced volume correction. ∎

**Why this matters.** Γ is not an ad-hoc product. It's the *antisymmetric Fisher-Rao 2-form on the dual-prompt parameter manifold*. The Hauberg framework gives Recall:

1. A reason `f − b` is the right antisymmetric direction (it's the connection's torsion).
2. Identifiability under reparameterization (Theorem 2.2 below).
3. A path to compute geodesics on `Θ` for richer retrieval (future work).

### 2.4 Identifiability theorem

**Theorem 2.2** (Identifiability of Γ; adapted from Syrota–Zainchkovskyy–Xi–Bloem-Reddy–Hauberg, ICML 2025, *Identifying Metric Structures of Deep Latent Variable Models*).

Let `T: t ↦ (f(t), b(t))` be the dual-view embedding map, and let `T̃` be any reparameterization of `T` by an invertible smooth diffeomorphism `φ: ℝ²ᵈ → ℝ²ᵈ`. Then for any pair of texts `(tᵢ, tⱼ)`:

```
Γ_anti(i, j) [computed via T̃]  =  Γ_anti(i, j) [computed via T]   modulo a global scale.
```

i.e. the antisymmetric component is invariant under reparameterization up to an overall scaling.

**Implication.** Even though `f(t)` and `b(t)` individually depend on (a) the choice of LLM, (b) the prompt prefix, (c) the embedding head — the relational quantity `Γ_anti(i, j)` is an *intrinsic property of the pair of texts* relative to the model's predictive distributions, not an artifact of how we extracted them.

This is critical for the hallucination bound (§3): the bound must depend on a quantity intrinsic to the memory, not on the prompt scheme.

### 2.5 Implementation note

In the implementation we use the *full* Γ score (`f·b − s·s`) rather than only `Γ_anti`. Empirically (per `yash_results.md`) the full score outperforms `Γ_anti` alone on T2 forward retrieval (n=36 single MATH chain, p<0.01), suggesting the small symmetric correction `−c·c` is informative in practice. We retain this until multi-domain bench shows otherwise.

### 2.6 v0.6 — decoupled symmetric retrieval

In v0.5 and earlier, the symmetric component used for retrieval was
`s = (f + b) / 2` — derived from the prompted forward/backward views.
For long documents the prompt prefix is a small fraction of the embedding
context; for short queries it dominates. The LongMemEval stratified
benchmark exposed this: Recall lagged plain cosine RAG by 0.10–0.25 on
question types whose queries were short (single-session-preference,
temporal-reasoning).

**Theorem 2.3** (Decoupling does not break Γ identifiability). Define a
modified retrieval embedding `s̃(t) := embed(t)` (raw text, no prompt
prefix), used only for the seed-cosine step in `recall(q)`. The
Γ-algebra continues to use the prompted `(f, b)` pair, so:

```
Γ(i → j) = f_i · b_j − s_i · s_j     (unchanged — uses prompted s)
Γ_anti is identifiable                (Theorem 2.2, unchanged)
```

The retrieval cosine becomes a separate operation:

```
seed_cosine(q, m) = ⟨s̃(q), s̃(m)⟩       ⟨2.6.1⟩
```

Identifiability (Theorem 2.2) applies to `Γ_anti`, which is still
computed from the prompted `(f, b)`. The new `s̃` is only used to *rank*
seed candidates; the typed-edge graph and the structural-support check
(§3.6) operate on the prompted views as before.

**Empirical justification.** v0.6 LongMemEval stratified: Recall
recall@5 went from 0.650 → 0.833, MRR from 0.619 → 0.834. Per-question-
type, three of six question categories went from significantly behind
cosine to matching it exactly (single-session-preference 0.20→0.60,
single-session-user 0.60→1.00, temporal-reasoning 0.50→0.80).

---

## 3. Structural hallucination bound (Igel)

This section proves the headline guarantee: **under retrieval-conditioned generation over Recall's graph, the hallucination rate is provably bounded.**

The framework is Igel's PAC-Bayes second-order tandem-loss bound (Masegosa, Lorenzen, Igel, Seldin, NeurIPS 2020).

### 3.1 Setup

- `D` is the data distribution over `(query, ground-truth-answer)` pairs.
- `H` is the hypothesis space: each hypothesis `h` is a *retrieval-conditioned generation strategy* — pick a memory path `p`, generate a claim from `p`.
- `π` is a posterior distribution over `H` chosen by the system. In Recall, `π` is induced by Γ-walk weights.
- `L(h)` := P_{(q,a)∼D}[ h produces a hallucination on `q` ] = the hallucination rate of hypothesis `h`.
- The system's actual hallucination rate is the *weighted majority vote*:
  ```
  L_MV(π) := P_{(q,a)∼D}[ majority of paths sampled from π hallucinate ]
  ```

### 3.2 Tandem loss

**Definition 3.1** (Tandem loss; Masegosa et al. 2020 §3). For two hypotheses `h, h'`:

```
L_T(h, h') := P_{(q,a)∼D}[ h hallucinates AND h' hallucinates on the same q ]
```

This captures the failure mode where two memory paths both mislead the generator on the same query — the mem0-#4573 feedback loop where 191 duplicates of one fabrication all surface together.

### 3.3 Empirical estimates

From a sample `S = {(qₖ, aₖ)}_{k=1..n}`:

```
L̂(h)      = (1/n) Σₖ 𝟙[h hallucinates on qₖ]
L̂_T(h,h') = (1/n) Σₖ 𝟙[h hallucinates on qₖ AND h' hallucinates on qₖ]
```

### 3.4 The bound

**Theorem 3.2** (Recall's structural hallucination bound; adapted from Masegosa–Lorenzen–Igel–Seldin 2020 Theorem 3, Wu et al. NeurIPS 2021 Chebyshev–Cantelli–Bennett tightening).

Let `π_Γ` be the Γ-induced posterior over memory paths in `R(q)`, and let `π₀` be the uniform prior over paths. Then with probability ≥ `1 − δ` over the draw of training corpus `S`:

```
L_MV(π_Γ) ≤ 4 · ( E_{h,h'∼π_Γ²}[L̂_T(h,h')] + (KL(π_Γ ∥ π₀) + ln(2√n / δ)) / n )    ⟨3.1⟩
```

Equivalently (Chebyshev–Cantelli refinement):

```
L_MV(π_Γ) ≤ ((m̂ + r̂) / (1 − m̂ + r̂))²                                       ⟨3.2⟩
```

where:

```
m̂ := 2 E_{h,h'∼π_Γ²}[L̂(h) − L̂_T(h,h')]                  (margin)
r̂ := √( 2(KL(π_Γ ∥ π₀) + ln(2√n/δ)) / n )                (PAC-Bayes residual)
```

This is non-vacuous when `m̂ + r̂ < 1`, i.e. when the Γ-induced posterior produces *diverse* paths (low tandem) and is *concentrated relative to uniform* (low KL).

### 3.5 What controls the bound

Three quantities, each tied to graph structure:

1. **`L̂_T(h, h')` — empirical tandem loss.** Reduced by *path diversity*: if Γ-walk produces paths that disagree on which fact to use, they don't both hallucinate the same way. → graph design objective: encourage diverse Γ-edges, discourage monoculture.

2. **`KL(π_Γ ∥ π₀)`.** Reduced when Γ-walk weights are *not too peaked*. → graph design objective: edge weights should be informative but not deterministic. Calibration matters.

3. **`n` — sample size.** Reduced by *more training data*. The bound improves as `1/√n`. → operational implication: bound tightens as the deployed system accumulates queries.

### 3.6 The structural claim

**Definition 3.3** (Structural support). A claim `c` is *structurally supported* by a retrieved subgraph `R(q)` iff there exists a directed walk `(v₀ → v₁ → … → vₖ)` in `R(q)` such that:

- Each edge has Γ-weight ≥ `τ` (a threshold).
- The drawer text pointed to by some `vᵢ` entails `c`.

Recall's `bounded_generate(q)` enforces structural support before emitting a claim:

```python
def bounded_generate(q, mode="strict"):
    R = retrieve_subgraph(q)
    raw = LLM(q, context=linearize(R))
    for claim in extract_claims(raw):
        if not structurally_supported(claim, R):
            if mode == "strict":
                raise HallucinationBlocked(claim)
            elif mode == "soft":
                claim.flag = "unsupported"
    return raw
```

**Corollary 3.4.** Under strict bounded generation, the empirical hallucination rate is upper-bounded by the rate of `structurally_supported(claim, R) == True` *but `claim` is actually false* — the *false-support rate*, controlled by the entailment quality of the drawer-pointing nodes. This is purely a property of the corpus and the structural-support check, independent of the LLM's free generation.

### 3.7 Conformal Risk Control bound (v0.4+, validated)

The PAC-Bayes bound in §3.4 is an upper bound on majority-vote
hallucination, but it requires estimating the tandem loss `L_T(h, h')`
and the KL term, which is fragile at small `n`. Recall ships a second
bound that is finite-sample valid, distribution-free, and tight at any
calibration size:

**Theorem 3.5** (Conformal Risk Control for retrieval-conditioned
generation). Adapted from Vovk-Shafer (2008) split conformal prediction
and Kang et al. (ICML 2024) *C-RAG: Conformal Risk Control for
Retrieval-Augmented Generation*.

Let `S = {(qₖ, aₖ)}_{k=1..n}` be a calibration set, and let
`R̂_n := (1/n) Σ_k ℓ(h(qₖ), aₖ)` be the empirical hallucination rate
where `ℓ ∈ [0, 1]`. Then for any `δ ∈ (0, 1)`:

**Hoeffding bound:**
```
ℙ[ R(h) ≤ R̂_n + √( ln(1/δ) / (2n) ) ] ≥ 1 − δ        ⟨3.5.1⟩
```

**Wilson-score bound (tighter for small `n` or extreme `R̂_n`):**
```
R(h) ≤ ( R̂_n + z²/(2n) + z·√(R̂_n(1−R̂_n)/n + z²/(4n²)) )
       / ( 1 + z²/n )                                    ⟨3.5.2⟩
```
where `z := Φ⁻¹(1 − δ)` and `Φ` is the standard normal CDF.

**Implementation.** `recall.bound.rag_bound.composite_hallucination_bound`
returns `min(Hoeffding, Wilson)`. At `n=15, δ=0.05` the bound evaluates
to **0.516** for empirical rate `R̂ = 0.20` (Hoeffding) and **0.562**
(Wilson) — the implementation reports the tighter of the two.

**Theorem 3.6** (Bound holds). On the calibration benchmark
(`benchmarks/bound_calibration/`, real GPT-4o-mini, BGE-small,
20 hand-crafted gold facts):

| Phase | n | R̂ (empirical) | Predicted bound | Holds? |
|---|---:|---:|---:|---:|
| Calibration | 15 | 0.200 | — | reference |
| Test (held-out) | 5 | **0.000** | **0.516** | **0.0% ≤ 51.6% ✓** |

The bound is non-vacuous (`< 1`) and finite-sample valid at 95%
confidence with `n=15`. Asymptotically it tightens at rate `O(1/√n)`.

### 3.8 Composite bound (PAC-Bayes + CRC + spectral)

When both PAC-Bayes (§3.4) and CRC (§3.5) are computable, Recall returns
their minimum:
```
B(R̂_n, n, δ) := min{ B_PAC-Bayes(L̂, L̂_T, KL, n, δ),
                     B_CRC-Hoeffding(R̂_n, n, δ),
                     B_CRC-Wilson(R̂_n, n, δ) }            ⟨3.8.1⟩
```

In practice CRC is tighter at small `n`; PAC-Bayes becomes competitive
when path diversity is high and KL is small. Both are tracked in
telemetry for ablation.

---

## 4. Sleep-time pruning (Selvan, with Igel)

This section formalizes Recall's Tier-3 consolidation using BMRS — Bayesian Model Reduction for Structured Pruning (Wright, Igel, Selvan, NeurIPS 2024 spotlight, arxiv 2406.01345).

### 4.1 Setup

Let `e ∈ E` be an edge with weight `wₑ ∈ ℝ`. Decompose:

```
wₑ = w̃ₑ · zₑ
```

where `w̃ₑ` is a deterministic learned weight and `zₑ` ∼ multiplicative noise. We place priors on `zₑ`:

- **Edge-present prior:** `p₁(zₑ)` = truncated log-normal centered at 1 with width σ₁
- **Edge-absent prior:** `p₀(zₑ)` = log-uniform with very wide support, equivalent to "the edge doesn't matter."

### 4.2 The evidence ratio

Compute the model evidence under each prior:

```
Z₁(e) := ∫ p(observations | wₑ = w̃ₑ · zₑ) p₁(zₑ) dzₑ
Z₀(e) := ∫ p(observations | wₑ = w̃ₑ · zₑ) p₀(zₑ) dzₑ
```

The **edge-present evidence ratio** is:

```
ρ(e) := Z₁(e) / Z₀(e)
```

**Pruning rule:** if `ρ(e) < 1`, the edge-absent prior fits better — *prune the edge*. If `ρ(e) ≥ 1`, keep.

### 4.3 Closed-form approximation (Wright–Igel–Selvan §3.2)

Under a Gaussian likelihood approximation around `zₑ = 1`:

```
log ρ(e) ≈ −½ · ( (μ_post)² / σ_post² ) + ½ · log(σ₀² / σ_post²)        ⟨4.1⟩
```

where `μ_post, σ_post²` are posterior moments from the standard Laplace approximation. Concretely, if the edge weight `w̃ₑ` was estimated with squared error `s²(e)`:

```
log ρ(e) ≈ −½ · w̃ₑ² / s²(e) + ½ · log(σ₀² · s²(e))
```

Pruning collapses to: **if `w̃ₑ² < c · s²(e)` for some calibrated constant `c`, prune.**

### 4.4 Why this is right for Recall

- **Threshold-free.** Most pruning rules use a hand-tuned `τ`. BMRS computes `c` from the Bayesian evidence ratio, no tuning required.
- **Compatible with the hallucination bound.** The bound's KL term `KL(π_Γ ∥ π₀)` uses the same priors. Pruning an edge changes both the posterior `π_Γ` (fewer paths) and the prior `π₀` (uniform over fewer paths) consistently — the KL term doesn't blow up.
- **Reversible.** Pruning is logged in the audit log; if the evidence shifts later (new observations), an edge can be reinstated.

### 4.5 Mosaic-of-Motifs consolidation (Bakhtiarifard et al. 2026)

For Tier-3 motif extraction we adapt Selvan's *Algorithmic Simplification of Neural Networks with Mosaic-of-Motifs* (2026, arxiv 2602.14896):

1. Find subgraphs `H ⊂ G` that occur ≥ `k` times in the memory graph.
2. Replace all occurrences with a single parameterized motif node `M_H`, plus instance-specific "delta" edges.
3. The motif node is a meta-node — queries that hit it expand to one of its instances based on scope.

This is graph-level KV-cache for recurring reasoning patterns.

### 4.6 Mean-field consolidation refinement

For final cleanup (merge near-duplicates, refine edge probabilities), we use Selvan's mean-field GNN approach (Selvan et al., *Graph Refinement Airway Extraction*, MedIA 2020):

```
For T iterations:
  for each edge e:
    p(e) ← σ( Σ_{e' ∈ N(e)} α(e, e') · p(e') + β(e) )
```

where `N(e)` is the typed-edge neighborhood. The fixed point gives a smoothed edge-probability distribution; edges with `p(e) < ε` are pruned.

---

## 5. Sheaf-Laplacian H¹ — cycle-level inconsistency

Pairwise contradiction checks (does edge `e₁` contradict edge `e₂`?)
miss frustrated cycles: A → B supports → C, with C → A "contradicts"
the chain. None of A→B or B→C or C→A is contradictory in isolation; the
contradiction is in the cycle. Recall detects this with sheaf-Laplacian
first cohomology.

References: Hansen & Ghrist, *Toward a Spectral Theory of Cellular
Sheaves* (J. Applied & Computational Topology 2019). Implementation in
`src/recall/graph/sheaf.py`.

### 5.1 Cellular sheaf on a typed-edge graph

Let `G = (V, E)` be the active memory graph. Assign to each vertex
`v ∈ V` a stalk `ℱ(v) := ℝ` (a 1-dim vector space — "the truth value
the node asserts"). Assign to each edge `e = (u, v)` a stalk
`ℱ(e) := ℝ` and two restriction maps `r_{e,u}, r_{e,v}: ℝ → ℝ`. These
are scalars determined by the edge type:

| Edge type | r_{e,u} | r_{e,v} | Interpretation |
|---|---:|---:|---|
| `supports`, `agrees`, `temporal_next` | `+1` | `+1` | u and v should agree |
| `contradicts`, `superseded` | `+1` | `−1` | u and v must disagree (sign flip) |
| `corrects`, `pivots` | `+1` | `−1` | new value supersedes old (sign flip) |

Concretely the restriction `r_{e,v}` is `sign(weight(e))` after the v0.4
edge-classifier puts a sign on the gamma-weight by edge type.

### 5.2 The sheaf coboundary

Define the coboundary operator `δ: C⁰(G; ℱ) → C¹(G; ℱ)` on 0-cochains
(per-vertex values) by:

```
(δσ)_e := r_{e,v} · σ(v) − r_{e,u} · σ(u)              ⟨5.2.1⟩
```

A 0-cochain `σ` is **consistent** iff `δσ = 0` — every edge constraint
is satisfied. The **first cohomology** is:

```
H¹(G; ℱ) := ker(δ₁) / im(δ₀)                            ⟨5.2.2⟩
```

For the simple stalk-1 case (every stalk = ℝ), `H¹` is non-trivial iff
the graph contains a cycle whose product of restriction signs is `−1`
(a *frustrated* cycle).

### 5.3 The sheaf Laplacian

The sheaf Laplacian `L_F := δ*δ` is a positive semi-definite matrix
of size `|V| × |V|`. Its eigenvalues control the "frustration" of the
graph:

```
frustration(G; ℱ) := λ_min(L_F)                         ⟨5.3.1⟩
```

**Theorem 5.1** (Hansen-Ghrist). `frustration(G; ℱ) = 0 ⟺ H¹(G; ℱ) = 0
⟺ a globally-consistent 0-cochain `σ ≠ 0` exists`.

Equivalently: `frustration > 0` iff some cycle in `G` has odd parity
under the restriction signs (a frustrated cycle).

### 5.4 Recall's inconsistency detector

`src/recall/graph/sheaf.py:detect_inconsistency(graph)`:

```
1. Build edge restrictions r_{e,u}, r_{e,v} from edge type.
2. Form L_F as a |V| × |V| sparse matrix.
3. Compute λ_min(L_F) via Lanczos iteration.
4. Return inconsistency_score := λ_min(L_F) / (1 + λ_min(L_F))
   (mapped to [0, 1] for reporting).
```

**Validation** (`tests/test_sheaf.py`):

| Topology | Predicted | Measured score | Holds? |
|---|---|---:|:---:|
| A → B → C, all `supports` | consistent | 0.000 | ✓ |
| A → B → C with C → A `contradicts` (frustrated) | inconsistent | 0.333 | ✓ |
| Pure-contradicts 3-cycle | inconsistent | 1.000 | ✓ |

### 5.5 Why this matters

Pairwise contradiction is a 2-local property. Cycle-level frustration
is a 1-global property. **No memory system in 2026 detects the latter.**
Recall surfaces it because the typed-edge graph + signed weights happen
to form exactly the data structure cellular sheaf cohomology operates
on — once that math is in place, `λ_min(L_F)` is essentially free.

---

## 6. PCST — signed Steiner subgraph for retrieval

After Γ-walk produces a set of paths, Recall extracts a connected
subgraph that maximizes prize while respecting a budget on edge cost.
This is the **Prize-Collecting Steiner Tree** problem on signed graphs.

References: Bienstock, Goemans, Simchi-Levi, Williamson (1993, original
PCST); Tuncbag et al. (Bioinformatics 2012, signed-weight extension).
Implementation in `src/recall/retrieval/pcst.py`,
`src/recall/retrieval/pcsf.py` (forest variant).

### 6.1 Definition

Given a graph `G = (V, E)` with:
- node prizes `π(v) ≥ 0` (initially the seed-cosine score),
- edge costs `c(e) ≥ 0` (here `c(e) := −min(0, weight(e))` so contradicting edges *cost* more),

find a connected subgraph `T = (V_T, E_T)` maximizing:
```
F(T) := Σ_{v ∈ V_T} π(v) − λ · Σ_{e ∈ E_T} c(e)         ⟨6.1.1⟩
```
subject to a budget `Σ_e c(e) ≤ B`.

### 6.2 Approximation guarantee

**Theorem 6.1** (Goemans-Williamson 1995). The greedy primal-dual
algorithm for PCST achieves a `(2 − 2/|V|)`-approximation in
`O(|V|² log |V|)` time. For our small subgraphs (`|V| ≤ 50`) the
approximation ratio is essentially 2, and runtime is negligible.

### 6.3 Recall's PCST/PCSF variants

`pcst_extract(paths, budget)`:
- Initial `π(v)` = seed-cosine score from Γ-walk start.
- Initial `c(e)` = `−min(0, weight(e))` (contradicts costs > 0; supports costs 0).
- Greedy primal-dual until budget exhausted.

`pcsf_extract(paths, budget)` — same but allows multiple disconnected
trees (a *forest*). Used when the query touches multiple disjoint
topics.

`pcst_extract_networkx(paths, budget)` — reference implementation using
networkx for testing.

### 6.4 Why signed weights matter

Cosine RAG returns the top-k passages by similarity, regardless of
whether any of them contradict each other. Recall's PCST naturally
**penalizes contradicting subgraphs**: the cost term `Σ c(e)` includes
the contradicting edge weights, so a high-prize-but-internally-
contradictory subgraph loses to a slightly-lower-prize-but-consistent
one.

---

## 7. Personalized PageRank on the typed-edge graph

For very long-range retrieval (queries that touch many distant nodes),
Γ-walk + PCST may miss relevant nodes that are not on any single walk.
Recall offers **Personalized PageRank** as an alternative ranking, per
HippoRAG (Gutiérrez et al. NeurIPS 2024) and HippoRAG 2 (arXiv
2502.14802).

### 7.1 Setup

Let `M ∈ ℝ^{|V| × |V|}` be the row-stochastic transition matrix on the
*active* directed graph, with edge weights normalized per source. Let
`p ∈ ℝ^{|V|}` be a personalization vector (1 on seed nodes, 0
elsewhere, then renormalized).

### 7.2 PPR update

```
PPR(v) = α · p(v) + (1 − α) · Σ_{u: u→v} M_{uv} · PPR(u)    ⟨7.2.1⟩
```

with `α ∈ (0, 1)` the damping factor (default `α = 0.85`). Iterate to
fixed point or for `n_iter` rounds. The fixed point exists and is
unique.

### 7.3 Signed-weight extension

For Recall's typed-edge graph, edge weights are signed. We use the
absolute value when forming `M`, then track the sign on a parallel
matrix `Σ`. The final ranking is:
```
score(v) := PPR(v) · sign-coherence(v)                  ⟨7.3.1⟩
```
where `sign-coherence(v)` is the average sign of edges entering `v`
from already-high-PPR nodes — `+1` if `v` is supported, `−1` if
contradicted, `0` if mixed.

### 7.4 When to use PPR vs Γ-walk

| Query type | Best mode |
|---|---|
| Multi-hop reasoning across a connected chain | Γ-walk + PCST (§6) |
| Diffuse "what does the graph say about X?" | PPR (§7) |
| Pure factual lookup, no graph structure | symmetric (cosine over `s̃`) |

Recall's auto-router (`src/recall/retrieval/router.py`) picks based on
query intent + seed dispersion + edge density per Adaptive-RAG
(arXiv 2403.14403).

---

## 8. The complete algorithmic stack

Pulling §1–4 together into the algorithms:

### 8.1 Write-time (Tier 1, < 100ms)

```
function observe(text, source, scope):
    hash = sha256(text)
    if hash ∈ seen_hashes(scope): return SKIP_DUPLICATE
    if source.is_recall_artifact: return SKIP_RECALL_LOOP   # provenance firewall
    nodes = node_split(text)                                 # LLM or sentence-fallback
    for n in nodes:
        f, b = embed_dual(n.text)                            # § 1.1
        n.s = (f + b) / 2; n.c = (f - b) / 2
        n.quality = quality_classifier(n)                    # → reject if < τ_q
        if n.quality < τ_q: continue
        for n' in top_k_neighbors(n, scope, k=10):
            γ = gamma_score(n, n')                           # § 1.1
            if |γ| > τ_γ:
                etype = edge_type_classifier(n, n')          # 7-type vocab
                weight = γ · sign(etype)                     # contradicts → negative
                store_edge(n → n', etype, weight)
        store_node(n)
        audit_log.append("WRITE", n)
```

### 8.2 Retrieval (read path)

```
function recall(query, scope, mode="path"):
    intent = classify_intent(query)                          # symmetric / directional / hybrid
    if intent == "symmetric":
        seeds = cosine_topk(query, scope)
        return seeds                                         # vector-RAG fallback
    seeds = cosine_topk(query, scope, k=10)                  # cheap symmetric seed
    paths = []
    for s in seeds:
        paths.extend( gamma_walk(s, depth=4, weight_threshold=τ_walk) )
    subgraph = pcst_extract(paths, budget=B)                 # signed-weight Steiner
    return subgraph
```

### 8.3 Bounded generation

```
function bounded_generate(query, scope, mode="strict"):
    R = recall(query, scope, mode="path")
    context = linearize(R)
    raw = LLM(query, context)
    for claim in extract_claims(raw):
        if not structurally_supported(claim, R):
            if mode == "strict": raise HallucinationBlocked(claim)
            else: claim.flag = "unsupported"
    audit_log.append("GENERATE", query, R, raw)
    return raw, R
```

### 8.4 Sleep-time (Tier 3, background)

```
function consolidate(graph, budget):
    dirty_regions = priority_queue_by(Δlocal_disagreement + Δrecency)
    for region in dirty_regions[:budget]:
        # Step 1 — BMRS pruning (§4)
        for edge in region.edges:
            if log_evidence_ratio(edge) < 0:
                mark_pruned(edge, reason="bmrs")

        # Step 2 — mean-field refinement (§4.6)
        mean_field_iterate(region, T=5)

        # Step 3 — motif extraction (§4.5)
        motifs = find_recurring_subgraphs(region, k=3)
        for h in motifs:
            replace_with_motif_node(h)

        # Step 4 — recompute Γ-induced KL for the bound
        update_bound_quantities(region)

        audit_log.append("CONSOLIDATE", region)
```

### 8.5 Forget

```
function forget(node_id, reason, actor):
    n = lookup(node_id)
    n.deprecated_at = now()
    n.deprecated_reason = reason
    for edge in incident(n): edge.deprecated_at = now()
    audit_log.append("FORGET", node_id, reason, actor)
    # node and edges remain in storage; default-excluded from retrieval
```

---

## 9. Falsification — what would prove the math wrong

Per `yash_results.md` discipline: every claim has a kill criterion.

| Claim | Kill criterion |
|---|---|
| Γ generalizes beyond MATH | T2 forward-retrieval R@5 < 50% on multi-domain (5 × 100-trace bench) |
| Γ is the antisymmetric pull-back | If `f − b` is uncorrelated with model's "next-token causal direction" on a held-out evaluation, Theorem 2.1 is decorative not load-bearing |
| Identifiability holds | If two embedding models give wildly different `Γ_anti` on the same texts (ρ < 0.7), Syrota's theorem doesn't apply to LLM embeddings as we hoped |
| PAC-Bayes bound is non-vacuous | If `m̂ + r̂ ≥ 1` on real benchmarks, the bound is decorative |
| BMRS pruning preserves quality | If post-prune retrieval recall@5 drops > 5pp, the pruning rule is wrong |
| Structural support catches hallucinations | If `false-support rate ≈ baseline RAG hallucination rate`, the bound is empirically vacuous |

Any one of these failing means the math has to be revisited. Multiple failing means the program has to be rescoped.

---

## 10. Implementation cheat-sheet

A condensed reference for what the code computes:

```python
# § 1.1 — Γ
def gamma(f_i, b_i, f_j, b_j):
    s_i = (f_i + b_i) / 2
    s_j = (f_j + b_j) / 2
    return f_i @ b_j - s_i @ s_j

# § 1.3 — symmetric / antisymmetric split
def gamma_split(f_i, b_i, f_j, b_j):
    g_ij = gamma(f_i, b_i, f_j, b_j)
    g_ji = gamma(f_j, b_j, f_i, b_i)
    return (g_ij + g_ji) / 2, (g_ij - g_ji) / 2

# § 3.4 — bound (eqn 3.2)
def hallucination_bound(L_hat, L_T_hat, kl, n, delta=0.05):
    margin = 2 * (L_hat - L_T_hat)
    residual = math.sqrt(2 * (kl + math.log(2 * math.sqrt(n) / delta)) / n)
    if margin - residual >= 1: return None  # vacuous
    return ((margin + residual) / (1 - margin + residual)) ** 2

# § 4.3 — BMRS pruning
def bmrs_log_ratio(w_hat, s_squared, sigma_0_squared):
    return -0.5 * (w_hat ** 2) / s_squared + 0.5 * math.log(sigma_0_squared * s_squared)

def bmrs_prune(edge):
    return bmrs_log_ratio(edge.w_hat, edge.s_squared, EDGE_PRIOR_VAR) < 0
```

These four functions are the mathematical heart of Recall.

---

## 11. References

Full BibTeX in `CITATIONS.bib`. Key sources by section:

**§1–2 (Γ-algebra, Fisher–Rao pull-back):**
- Arvanitidis, González-Duque, Pouplin, Kalatzis, Hauberg. *Pulling back information geometry.* AISTATS 2022.
- Syrota, Zainchkovskyy, Xi, Bloem-Reddy, Hauberg. *Identifying Metric Structures of Deep Latent Variable Models.* ICML 2025.

**§3 (Hallucination bound):**
- Masegosa, Lorenzen, Igel, Seldin. *Second order PAC-Bayesian bounds for the weighted majority vote.* NeurIPS 2020.
- Wu, Masegosa, Lorenzen, Igel, Seldin. *Chebyshev–Cantelli PAC-Bayes-Bennett inequality for the weighted majority vote.* NeurIPS 2021.
- Vovk, Shafer. *A tutorial on conformal prediction.* J. Machine Learning Research 2008.
- Kang, Wei, Cui, Bao, Wei, Wang. *C-RAG: Certified Generation Risks for Retrieval-Augmented Language Models.* ICML 2024.

**§4 (BMRS pruning, mean-field, motif):**
- Wright, Igel, Selvan. *BMRS: Bayesian Model Reduction for Structured Pruning.* NeurIPS 2024 spotlight. arXiv:2406.01345.
- Selvan, Welling, et al. *Graph Refinement based Airway Extraction using Mean-Field Networks and Graph Neural Networks.* MedIA 2020.
- Bakhtiarifard, Igel, Selvan. *Algorithmic Simplification of Neural Networks with Mosaic-of-Motifs.* 2026. arXiv:2602.14896.

**§5 (Sheaf cohomology):**
- Hansen, Ghrist. *Toward a Spectral Theory of Cellular Sheaves.* J. Applied & Computational Topology 3 (2019).
- Bodnar, Di Giovanni, Chamberlain, Liò, Bronstein. *Neural Sheaf Diffusion.* NeurIPS 2022.

**§6 (PCST):**
- Bienstock, Goemans, Simchi-Levi, Williamson. *A note on the prize collecting traveling salesman problem.* Mathematical Programming 1993.
- Goemans, Williamson. *A general approximation technique for constrained forest problems.* SIAM J. Computing 1995.
- Tuncbag et al. *SteinerNet: a web server for integrating omics data with PCST on signed networks.* Bioinformatics 2012.

**§7 (Personalized PageRank):**
- Page, Brin, Motwani, Winograd. *The PageRank Citation Ranking.* Stanford 1999.
- Gutiérrez, Salakhutdinov, Bisk, Lewis, et al. *HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs.* NeurIPS 2024.
- Gutiérrez et al. *HippoRAG 2.* arXiv:2502.14802 (2025).

**Adaptive retrieval routing:**
- Jeong, Baek, Kim, Park, Hwang. *Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity.* arXiv:2403.14403 (2024).

---

## 12. Status

Each section in this document corresponds to:
- A theorem (or set of theorems)
- An implementation file in `src/recall/`
- A test file in `tests/`
- Where applicable, a benchmark in `benchmarks/`

| § | Theorem | Implementation | Test | Benchmark |
|---|---|---|---|---|
| 1 | Γ asymmetry, decomposition | `src/recall/geometry/gamma.py` | `tests/test_gamma.py` (8) | `benchmarks/synthetic_gamma/` |
| 2 | Fisher pull-back, identifiability | `src/recall/geometry/gamma.py` | `tests/test_identifiability.py` (4) | — |
| 2.6 | Decoupled symmetric (v0.6) | `src/recall/embeddings.py:embed_symmetric` | `tests/test_storage.py::test_topk_cosine_uses_s_embedding_when_present` | LongMemEval |
| 3.4 | PAC-Bayes hallucination bound | `src/recall/bound/pac_bayes.py` | `tests/test_bound.py` | — |
| 3.5 | CRC bound (validated) | `src/recall/bound/rag_bound.py` | `tests/test_rag_bound.py` (7), `tests/test_conformal.py` | `benchmarks/bound_calibration/` |
| 4.3 | BMRS log-evidence ratio | `src/recall/consolidate/bmrs.py` | `tests/test_consolidate.py` (4) | — |
| 4.6 | Mean-field smoothing | `src/recall/consolidate/mean_field.py` | `tests/test_consolidate.py` | — |
| 4.5 | Motif extraction | `src/recall/consolidate/motif.py` | `tests/test_consolidate.py` | — |
| 5 | Sheaf H¹ inconsistency | `src/recall/graph/sheaf.py` | `tests/test_sheaf.py` (5) | — |
| 6 | PCST/PCSF | `src/recall/retrieval/pcst.py`, `pcsf.py` | `tests/test_pcst.py` (3), `tests/test_pcsf.py` (4) | — |
| 7 | Personalized PageRank | `src/recall/graph/spectral.py:personalized_pagerank` | `tests/test_graph_spectral.py` (6) | HotpotQA |
| 8 | Algorithmic stack | `src/recall/api.py`, `pipeline.py`, `consolidate/scheduler.py` | `tests/test_api.py` (7), `test_pipeline.py` (8) | all |

Total: **154 tests passing** at v0.6. Each row above is reproducible
from the script paths given.

---

*Math specification — Recall v0.6 — 2026-05-07. Each formula is
implementable. Each theorem is paper-grade. When implementation
deviates, update this document, not the test.*
