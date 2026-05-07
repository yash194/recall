# Deployment Plan — Recall v0.4

> Three deployment modes, with concrete tech choices, isolation guarantees,
> failure-mode runbooks, and go/no-go checklists.

---

## Three deployment modes

### Mode 1 — Embedded (`pip install typed-recall`)

**For**: notebook research, single-user agent, on-device, air-gapped.

- **Runs**: in-process Python. `Memory(tenant="...", storage="sqlite://./recall.db")`.
- **Storage**: SQLite WAL. One file per tenant.
- **Auth**: none. Process-level trust boundary.
- **Latency**: observe p50 30ms / p99 120ms; recall p50 40ms / p99 200ms; bounded_generate p50 600ms / p99 2s (LLM-bound).
- **Cost**: zero infra. ~1 KB/node disk. LLM calls are user's own bill.
- **Scale ceiling**: ~500K nodes per tenant before brute-force cosine in `topk_cosine` needs replacement with hnswlib/FAISS.
- **Install**: `pip install typed-recall` (or `pip install typed-recall[embed-bge,llm-openai]`).

### Mode 2 — Self-hosted server (Docker)

**For**: a startup running Recall in their own VPC, one team behind a single API key, regulated industry.

- **Runs**: one container = `uvicorn recall.server:app` + sidecar `recall-consolidator` worker. Optional Postgres container.
- **Storage profiles** (env-flagged):
  - `RECALL_STORAGE=sqlite` (single-tenant, hobby) — bind-mount `/data`.
  - `RECALL_STORAGE=postgres` (multi-tenant within an org) — `DATABASE_URL` points at Postgres + pgvector with one schema per tenant.
- **Auth**: API key per tenant via `RECALL_KEYS=key1:tenantA,key2:tenantB` env or OIDC bearer (`RECALL_OIDC_ISSUER`).
- **Latency**: observe p50 80ms / p99 300ms; recall p50 120ms / p99 500ms; bounded_generate p50 700ms / p99 2.5s.
- **Cost**: 1 small VPS ($20–80/mo) handles <50 tenants. ~10 MB DB / 1K observations per tenant. LLM cost flows to configured provider.
- **Scale**: vertical to ~500 tenants on a 4-vCPU/16 GB box; horizontal via consistent-hash sharding on `tenant_id`.
- **Install**: `docker run ghcr.io/recall/server:0.4` or `docker compose up`.

### Mode 3 — Recall Cloud (recall.dev managed)

**For**: serverless functions, agent SaaS apps, anyone who'd rather not run infra.

- **Runs**: Kubernetes (GKE Autopilot recommended). Stateless API, stateless workers, stateful Postgres (per-tenant schema), async consolidator workers, S3 cold storage.
- **Storage**: Postgres-per-tenant-schema with pgvector HNSW. Hourly `pg_dump` to S3 with object lock. 30-day hot retention. Daily `pg_basebackup` to S3 versioning + object lock.
- **Auth**: OIDC (Auth0) for dashboard → JWT for API. Programmatic clients use API keys (prefix `rk_live_`) hashed with Argon2 in `api_keys` table. Auth middleware resolves key → `tenant_id`, cached 60s.
- **Latency** (regional, warm cache): observe p50 120ms / p99 400ms; recall p50 150ms / p99 600ms; bounded_generate p50 1.2s / p99 3.5s.
- **Cost shape per tenant/month**: storage $0.01/MB on RDS gp3 + S3 cold $0.001/MB; vector index RAM $0.20 per 100K nodes; LLM passthrough at +20% margin.
- **Free tier**: 10K observations + 1K bounded_generates / month.
- **Scale**: API/workers horizontal (HPA on RPS + p99 latency). Postgres tier-sharded — free/hobby on shared instance, pro/enterprise dedicated.

---

## Cloud architecture (managed mode)

```
                ┌────────────┐
   Client ──TLS─▶  Cloudflare (DDoS, WAF, geo)
                └────┬───────┘
                     ▼
                ┌────────────┐
                │  Envoy     │  rate-limit (1K rps/tenant, burst 50)
                │  gateway   │  JWT/API-key validation, tenant injection
                └────┬───────┘
                     ▼
              ┌──────────────┐         ┌──────────────────┐
              │  api-worker  │ ──RPC──▶│  llm-router      │ ──▶ OpenAI / Anthropic / local
              │  (FastAPI)   │         │  (cost meter)    │
              │  HPA 3–50    │         └──────────────────┘
              └──┬─────────┬─┘
                 │         │
                 ▼         ▼
       ┌────────────┐  ┌─────────────┐
       │  Postgres  │  │  Redis      │  cache: tenant routes, JWT, idempotency
       │  +pgvector │  │  (Upstash)  │
       │  per-tier  │  └─────────────┘
       └─────┬──────┘
             │ logical-replication
             ▼
       ┌────────────┐         ┌────────────────┐
       │  S3 cold   │ ◀──────│ consolidator   │  K8s CronJob, fan-out per
       │  pg_dump   │  reads │  worker pool   │  dirty-tenant queue (SQS)
       └────────────┘        └────────────────┘
```

### Stack choices

| Layer | Choice | Why |
|---|---|---|
| Edge | Cloudflare → Envoy | DDoS + WAF + JWT auth + rate-limit. Standard. |
| Compute | K8s (GKE Autopilot) | HPA, pod disruption budgets, real CronJob primitive |
| Backend store | **Postgres + pgvector**, NOT KuzuDB | Managed offering, PITR, mature multi-tenant story; Kuzu is single-writer file-based. Edge counts at Recall's scale (millions, not billions) fit Postgres comfortably. |
| Cache | Upstash Redis | Tenant-route cache, JWT cache, idempotency keys |
| Cold storage | S3 with object lock + KMS-CMK | Compliance-grade backup |
| Queue | SQS | Per-tenant consolidation jobs |
| Observability | Prometheus + Grafana + Loki + Tempo | OTel everywhere |
| Auth | Auth0 (OIDC) + Argon2-hashed API keys | Standard |

---

## Per-tenant isolation — defense in depth

Three concentric layers, all enforced — single-layer breakage is an
existential bug.

1. **Network/auth (gateway)**: Envoy maps API key → `tenant_id` and
   stamps it as an immutable `x-recall-tenant` header. The body field
   `tenant` in `server.py` is overwritten server-side; if header and
   body disagree, the request is 403'd and audit-logged.

2. **Application (per-tenant Memory cache)**: the `_memories: dict[str,
   Memory]` cache becomes a TTL'd LRU keyed by header tenant; every
   request goes through `get_memory(request.state.tenant)`. Handler
   functions cannot construct a different Memory.

3. **Database (row + schema)**: each tenant lives in its own Postgres
   schema (`tenant_<uuid>`). The Memory's storage connection sets
   `SET search_path TO tenant_<uuid>` once on connect. Even if a query
   forgets a `WHERE tenant = ?`, it cannot reach another tenant's data
   because the tables aren't in scope. Belt-and-braces: keep the
   existing `tenant TEXT NOT NULL` columns and add a
   `CHECK (tenant = current_setting('recall.tenant_id'))` constraint.

Audit: every cross-tenant access attempt writes to a global
`security_events` table and pages on-call.

---

## Failure modes & recovery

### Consolidator crash mid-pass

The existing `Consolidator.run()` is single-shot. Wrap it in a driver
that processes one region at a time and writes
`(tenant, region_id, completed_at, last_audit_seq)` after each region
commits. On worker restart, SQS redelivers the message; the driver
skips already-completed regions. Worst case: a half-pruned region; BMRS
is monotonic in evidence, so re-running converges.

### LLM provider down during bounded_generate

Add 3s timeout, then failover to secondary provider via `LLMRouter`. If
both fail, return HTTP 503 with `Retry-After: 30`. **Do not write the
audit GENERATE entry** (the call effectively didn't happen). Retrieval
results are returned in the 503 body so the client can render "we found
context but couldn't generate" — the retrieval was free.

### Backup / restore per tenant

- Hourly `pg_dump --schema=tenant_<uuid>` to S3 (per-tenant prefix,
  AES-256 SSE, KMS-CMK for enterprise)
- Restore: `recall-cloud restore --tenant <id> --at 2026-05-07T14:00Z`
  creates a `tenant_<uuid>_restore` schema, applies dump, atomically
  swaps via `ALTER SCHEMA RENAME`

### Point-in-time recovery (PITR) of audit log

Combination of: (a) hourly `pg_dump`, (b) WAL archiving to S3,
(c) `audit_log` JSONL export every 5 min via logical replication slot
gives **minute-granularity PITR**. The audit log is the source of truth
— if Postgres is corrupt, you can rebuild graph state by replaying
`OBSERVE` audit entries.

---

## Monitoring & SRE

| Metric | Source | Alert threshold |
|---|---|---|
| `recall_p99_recall_latency_ms{tenant}` | metrics.snapshot() | >800ms for 5min → page |
| `recall_p99_observe_latency_ms{tenant}` | same | >500ms for 5min → page |
| `recall_junk_rate{tenant}` | `nodes_rejected_quality / promoted+rejected` | >0.6 for 1h → ticket (model regression) |
| `recall_hallucination_block_rate{tenant}` | `hallucinations_blocked / bounded_generate_count` | >0.3 for 1h → ticket; >0.7 for 5min → page |
| `recall_consolidation_budget_remaining{tenant}` | `daily_budget - regions_processed` | <20% with 4h left → page |
| `recall_llm_cost_usd_per_min{tenant}` | sum llm_costs.est_usd | >$5/min for any tenant → throttle + page |
| `recall_audit_log_lag_seconds` | `now - max(timestamp)` from S3 export | >300s → page (compliance critical) |
| `recall_postgres_connection_saturation` | pg `numbackends / max_connections` | >0.8 → page |

**Alerting policy**: PagerDuty escalation: warn → Slack #recall-ops;
page → on-call primary; if unacked 15min → secondary; if unacked 30min → CTO.

**Error budget**: 99.5% monthly availability for `recall.observe`,
99.0% for `bounded_generate` (LLM-dependent).

**Cost dashboard per tenant**: Grafana panel showing storage MB, vector
index MB, daily LLM $, daily compute $, all rolled to
`revenue - cost = margin` per tenant. Negative margin for 7 days →
auto-bump off free tier.

---

## Data lifecycle

- **Retention**: default infinite (memory IS the product). Per-tenant
  override via `retention_days` config; daily job runs `mem.forget(...)`
  on nodes whose `created_at < cutoff` AND have `quality_status='retired'`.
  Promoted nodes never auto-forgotten. Audit entries never deleted
  (legal record); age out to S3 Glacier after 1y.

- **Export/import**: `recall export --tenant <id> --out <path>` writes
  `audit.jsonl`, `nodes.jsonl`, `edges.jsonl`, `drawers.jsonl`. Import
  is the inverse with strict tenant remap.

- **GDPR deletion**: `DELETE /v1/tenants/<id>?confirm=<id>` →
  enqueues `tombstone` job → `DROP SCHEMA tenant_<uuid> CASCADE` +
  delete S3 backups (override object lock with legal-hold-release) +
  delete audit S3 export + delete `api_keys` rows. Emits final
  `TENANT_DELETED` entry to compliance log retained 7 years. **Hard SLA:
  30 days from request to full erasure.**

---

## Go/no-go checklists

### Embedded — ship anytime once:
- [x] Tests >70% pass on Python 3.10–3.12 (currently 144/144)
- [x] `pip install typed-recall[embed-bge,llm-openai]` clean install on macOS+Linux
- [x] README quickstart works in <5 lines
- [ ] No required env vars

### Self-hosted server — ship when:
- [ ] Docker image <500MB
- [ ] Postgres adapter implements Storage protocol
- [ ] 144-test parity run passes vs SQLite
- [ ] OIDC auth path tested with Keycloak
- [ ] `docker compose up` brings up all services in <60s
- [ ] Restore-from-backup runbook executed end-to-end in staging

### Recall Cloud — ship when:
- [ ] Per-tenant schema isolation passes red-team test
- [ ] `recall_audit_log_lag_seconds` SLO held for 7 days in staging
- [ ] Billing pipeline reconciles to <0.1% of provider invoice
- [ ] GDPR delete tested end-to-end with verification
- [ ] On-call runbook covers all 8 alerts above
- [ ] Status page live
- [ ] ToS + DPA reviewed by counsel
- [ ] Rate-limit DoS test passed (10K rps from 100 IPs without affecting other tenants)

---

## Critical files for implementation

- `src/recall/server.py` — needs middleware to extract `tenant` from JWT/API key header
- `src/recall/core/storage.py` — needs Postgres adapter implementing the Storage protocol
- `src/recall/api.py` — should accept `Memory` from request scope, not construct in handler
- `src/recall/consolidate/scheduler.py` — needs checkpointed driver for SQS-backed processing
- `src/recall/audit/log.py` — already perfect; reuse for tenant-level deletion audit
