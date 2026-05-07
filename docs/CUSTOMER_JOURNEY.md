# Customer Journey — exactly how Recall is used and felt

> Four personas, exact play-by-play of how each plugs in, what they see in
> the first 5 minutes, the first day, the first week, and the moment they
> *feel* the improvement.

---

## Persona 1 — The indie dev building an AI agent

**Background**: Maya, 28, building a customer-support chatbot for a SaaS startup. Uses OpenAI/Claude API, FastAPI backend. Ships every 2 weeks.

### Day 0 — discovery (5 min)

She lands on `recall.dev` after a teammate forwards the HN post. The hero
shows a 15-second GIF: split-screen of two agents asked the same question
30 days into a conversation. The left one (vanilla) says "I don't know
which queue you're using." The right (Recall) cites the exact decision
made on day 12 and links the original message. **The "feels different"
moment lasts about 4 seconds — she sees the path graph render.**

### Minute 0–5 — install

```bash
pip install recall
```

```python
from recall import Memory
mem = Memory(tenant="acme_support")

# In her existing turn handler:
mem.observe(user_msg, agent_msg, scope={"customer_id": cust.id})

# When generating a response:
ctx = mem.recall(user_question, scope={"customer_id": cust.id})
# pass ctx.subgraph_nodes into her existing prompt
```

Three lines. Her existing FastAPI endpoint compiles. Tests pass.

### Hour 1 — first integration

She runs the bot against a fresh customer transcript. The default
`HashEmbedder` works but is dumb — she sees that. README points her to
`pip install recall[embed-bge]` and `Memory(embedder=BGEEmbedder(...))`.
Now retrieval is real.

She tries the demo question: "what queue tech are we using?" 30 messages
into the conversation. Bot answers correctly with the exact original
sentence cited.

### Day 1 — felt improvement

A real customer hits a follow-up at 2pm: "is that still our recommended
approach?" Vanilla bot would have said "I'd need more context." Maya's
bot answers using the architectural decision recorded earlier in the
ticket. Customer is happy. Maya screenshots it for the team Slack.

**What she felt:** The agent stopped re-asking for context. She didn't
have to write a custom retrieval pipeline. The 3-line API saved her a
sprint of work.

### Week 1 — routine

She integrates `mem.forget(node_id, reason="customer corrected")` into
her admin UI. When a support agent corrects a fact, it sticks. Audit log
shows what was deleted.

She runs `mem.consolidate()` nightly. The graph stays bounded around
3,000 active nodes per customer despite hundreds of turns.

### Month 1 — substantial benefit

Mean handle time drops 18% (her metric, not ours). She cites Recall in
her sprint review as the win. The PM asks if it's the kind of thing they
should pay for. She says "the OSS is fine, but if we hit 50K customers
the managed tier looks worth it."

---

## Persona 2 — The Cursor / Claude Desktop / Cline power user

**Background**: Jordan, 31, full-stack engineer. Switches between Cursor,
Claude Desktop, and Cline daily. Frustrated that each tool has its own
"memory" silo.

### Day 0 — discovery (3 min)

Sees the launch on r/LocalLLaMA. The post says "MCP server — drops into
Claude Desktop / Cursor / Cline / Continue. One memory across every AI
tool you use."

### Minute 0–3 — install

```bash
claude mcp add recall -- uvx --from recall recall-mcp
```

That's it. Or in Cursor settings → MCP → add `uvx --from recall recall-mcp`.

### Minute 3–10 — first test

In Claude Desktop he says "we just decided to switch to Postgres LISTEN/
NOTIFY for the queue." Closes Claude. Opens Cursor. Asks "what queue
tech are we using?"

Cursor's reply: "You decided on Postgres LISTEN/NOTIFY based on a
discussion logged 4 minutes ago in Claude Desktop. Want me to check the
exact reasoning?"

**The "felt difference" — 8 seconds of jaw drop.** This has never worked
across two AI tools before in his life. The memory is shared because
both tools share the Recall MCP server.

### Hour 1 — depth

He asks Cursor "why did we pick that?" Cursor traverses the typed-edge
graph and returns: "trigger: PagerDuty alerts revealed message loss in
your prior implementation. pivot: discussed Redis Streams alternative
but rejected on operational complexity. decision: LISTEN/NOTIFY with
upper-bound retry semantics."

That's not in his note-taking. That's in the graph he built by talking
to two different AI tools.

### Day 1 — routine

He opens 3 tools simultaneously. They agree on his decisions because
they all read the same Recall MCP. He never re-explains his architecture.

### Week 1 — felt improvement

Friday afternoon, designer asks "what's your stack again?" He pulls up
Cursor and asks. The answer is right. He shares the trace (provenance:
which message in which tool on which day). Designer is delighted. The
memory is portable.

**What he felt:** The AI tools stopped being islands. He stopped being
the integration layer. The MCP install was trivial. The graph audit
makes him feel like he's actually controlling his AI stack instead of
being its janitor.

---

## Persona 3 — The personal-knowledge user

**Background**: Hiroshi, 35, researcher. Org-mode user for a decade.
Wants Recall as a personal "second brain" with reasoning paths — not
just full-text search.

### Day 0 — discovery (5 min)

Hears about Recall on the AI-PKM thread of r/Obsidian. Three sentences
in: "Recall stores typed edges between thoughts; you can ask `why did I
change my mind on X` and it walks the graph backward."

### Minute 0–5 — install

```bash
pipx install recall
recall me add "decided to migrate from Postgres LISTEN/NOTIFY to Redis Streams because of message loss"
recall me add "Redis Streams gives durability, validated in staging"
recall me ask "why redis?"
```

The CLI returns the connected thoughts with their roles tagged
(`decision`, `attempt`, `outcome`). Not a flat list.

### Hour 1 — bulk import

```bash
recall me ingest ~/notes/2026/
```

Pipes his org-mode notes into the graph. 1,200 paragraphs become 800
nodes (after quality gating filters meta-text).

```bash
recall me health
```

Output:
```
=== Memory health ===
  nodes: 800
  edges: 2400
  density: 3.0

=== Connectivity (spectral) ===
  λ_2 (spectral gap): 0.034
  Cheeger lower bound: 0.017

=== Topology (persistent homology) ===
  β_0 (components): 8
  β_1 (loops): 12

=== Curvature (Ollivier-Ricci) ===
  bottleneck edges: 19
  community edges: 2381
```

He can SEE his note structure for the first time. β_0 = 8 means his
notes break into 8 conceptual islands. He didn't know that.

### Day 1 — felt improvement

He asks `recall me ask "what changed about my view on REST APIs?"`. The
graph walks `corrects` and `pivots` edges from 2024 forward. He reads
a chronology of his own thinking. His old "search by keyword" workflow
returned nothing because the words "REST" and "GraphQL" appeared in
unrelated notes.

**What he felt:** The notes have geometry now, not just text. He sees
his own intellectual evolution as a graph. He tells his colleague it
"feels like Obsidian if Obsidian could reason."

### Week 1 — routine

He runs `recall me consolidate` weekly. BMRS prunes weak edges.
Sheaf inconsistency detector flags two contradictions in his notes
he didn't realize he held. He resolves them.

### Month 1 — substantial benefit

He writes a paper. The literature review section pulls out citations
in 30 minutes instead of 3 days because the graph already encoded
which sources he supported, contradicted, or pivoted on.

---

## Persona 4 — The enterprise team

**Background**: Alex, eng manager at a Series B fintech. 30-engineer
team. Considering AI agents for internal documentation but blocked by
hallucination liability (Air Canada / Cursor incidents).

### Week -2 — discovery (research call)

Procurement requirements: SOC 2 Type II, audit logs, BYOK, GDPR delete,
non-vacuous hallucination bound, on-prem option. Vendors evaluated:
Mem0, Cognee, Letta, Recall.

Recall stands out on three: (1) Apache-2.0 with public OSS-forever
commitment in GOVERNANCE.md, (2) cellular sheaf H¹ inconsistency
detector + Conformal Risk Control bound (the only vendor with a
*non-vacuous* mathematical hallucination guarantee), (3) self-hosted
Docker option with Postgres backend.

### Day 0 — pilot

```bash
docker compose up
```

`docker-compose.yml` brings up `recall-server`, `recall-consolidator`,
Postgres, Redis. 60 seconds, all healthy. They point their Letta agent
at it via the MCP server.

### Week 1 — pilot tests

Engineering lead runs the team's documentation through `recall ingest`.
2,800 nodes after junk gating. Quality classifier rejected 1,400 of
4,200 candidates (33% — fabricated profile claims auto-detected, no
LLM hallucination stored).

Audit log captures every retrieval. Compliance officer runs a sample
audit on 50 retrievals — every answer's provenance is reachable.

### Week 2 — felt improvement

CISO asks "how do we know it isn't hallucinating?" Lead shows the
`bound_value` returned with each generation. Conformal Risk Control bound
sits at 0.18 (vs. 1.0 vacuous on competitors). On 300 calibration
queries, empirical hallucination rate is 11.3% with bound 18.4% — both
non-vacuous.

CISO asks "what if engineer-A's notes leak to engineer-B?" Lead shows
the schema-isolation: per-tenant Postgres schemas with `SET search_path`.
Red team tries to read across; it fails at the database layer, not the
application.

### Month 1 — production rollout

Recall Cloud isn't ready (waitlist), so they self-host. Procurement
signs the Apache-2.0 + DPA addendum. Goes live for 50 internal users.

### Month 3 — substantial benefit

- Engineer onboarding time: 6 weeks → 4 weeks (they ask the bot, get a
  cited answer, audit-trail visible)
- Doc-search escalation rate: 30% → 8%
- Compliance: zero AI-generated hallucination incidents (vs. 3 in their
  prior LangChain-based pilot)

**What Alex felt:** The bound number stopped being a marketing claim and
became a procurement deliverable. The OSS-forever commitment in
GOVERNANCE.md unblocked their CTO. The audit log meant they could
defend an AI answer in front of a regulator if needed.

---

## What "feels like an improvement" means, distilled

For all four personas, the felt difference reduces to **three concrete
moments**:

1. **The "remembered" moment** — they say something on day N, and on day
   N+30 the AI brings it back unprompted, with provenance. Vanilla
   tools cannot do this; competing memory tools store it but
   typically can't surface the *why*.

2. **The "fix-and-forget" moment** — they correct the AI ("no, that
   project shipped in March, not January"), and the correction sticks.
   `mem.forget(...)` is one call; the audit log preserves what changed.

3. **The "audit" moment** — someone asks "why did the AI say X?" and the
   response is a graph: the path of typed edges from query to answer,
   with the original source messages cited. This is what makes
   Recall procurement-grade.

Each persona experiences these at different scales (Maya: customer
satisfaction, Jordan: workflow continuity, Hiroshi: intellectual
geometry, Alex: compliance defense). The *mechanism* is the same — typed
edges + bounded generation + audit trail.

---

## How Recall reaches each persona

| Persona | Discovery channel | Install path | Time-to-felt-improvement |
|---|---|---|---|
| Maya (indie dev) | HN, r/LocalLLaMA, Letta Discord | `pip install recall` | ~1 day |
| Jordan (AI-tool user) | r/LocalLLaMA, Cursor forum, Twitter | `claude mcp add recall` | ~10 minutes |
| Hiroshi (PKM user) | r/Obsidian, dev.to, Hacker News | `pipx install recall` + `recall me ingest` | ~1 hour |
| Alex (enterprise) | Cold inbound, conference, vendor eval | `docker compose up` | ~2 weeks (security review) |

Each path is in the launch playbook (LAUNCH.md). Each gets its own
section of the README so the persona finds their path within 5 seconds.
