# PromptFoundry

Continuous, evidence-driven discovery and production of **original** prompt products:
demand signals → scored opportunities → Prompt-IR → staged evaluation → governance gate →
human approval → publish → online experiment → learn or roll back.

Runs with **zero API keys and no network access**: the demo uses operator-uploaded CSV,
a fully mocked official-API connector, a deterministic model simulator and a mock
storefront. Adding provider keys upgrades the same code paths to real models.

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
make demo      # seed + full offline pipeline, ~15s
make test      # pytest
make lint      # ruff
make api       # http://127.0.0.1:8000/docs
```

## Why this exists

Most "prompt marketplace" automation is either a content mill (scrape demand, paraphrase
someone else's prompt, publish unattended) or a demo that never ships. PromptFoundry is
built the other way round:

| Risk | Control in this repo |
| --- | --- |
| Scraping / ToS violation | Connectors are official-API, upload or first-party only. Every source stores a `permission_basis`; `check_permission_basis()` refuses mismatches. There is no HTML fetcher, no headless browser, no CAPTCHA/paywall path. |
| Copying competitor prompts | Prompts are **synthesised from a specification (Prompt-IR)**, never templated from observed text. A similarity gate blocks cosine ≥ 0.92 or 5-gram Jaccard ≥ threshold against our catalog and a licence-cleared corpus. |
| Unreliable quality claims | Deterministic assertions first, calibrated LLM-judge second (trust = f(κ), halved below κ 0.4), bootstrap confidence intervals, 11 attack classes, holdout suite excluded from optimisation. |
| Regulated-domain harm | Medical/legal/financial/etc. are detected from the *task*, forced into assistive-only mode with disclaimers and refusal rules, and never auto-published. |
| Unattended mistakes | Publishing needs a human approval by default (separation of duties: the requester cannot approve). Kill switches, budget breaker, rollback with audit. |
| Runaway cost | All spend is AUD micros in a ledger; daily/weekly caps are enforced *before* each LLM call (HTTP 402 + alert). Staged evaluation caps spend per candidate. |

## The loop

```
official APIs / uploads / own analytics / synthetic      ← permission_basis recorded
        │
        ▼  normalise → embed → DBSCAN → topics (JTBD, buyer segment, drift)
opportunities   profit-first, uncertainty-aware, 90% bootstrap CI, ranked by lower bound
        │
        ▼  Prompt-IR (original by construction) → variants → 3-stage optimisation
product version  screen → offline → finalist (cost-capped, holdout never seen)
        │
        ▼  gate: policy → similarity/IP → evaluation → injection → risk → approval
human approval   (separation of duties, audited)
        │
        ▼  deterministic signed ZIP (per-file SHA-256, HMAC manifest) + listing
storefront       mock | Gumroad | Lemon Squeezy (official APIs only)
        │
        ▼  A/B or bandit with Wilson CIs, guardrails, attribution checks
learn or roll back
```

## Layout

| Path | Contents |
| --- | --- |
| `backend/` | FastAPI app, routers, request deps (RBAC, rate limits, audit), services |
| `core/` | settings, RBAC, errors, ids, artifacts, budget breaker, security |
| `db/` | SQLAlchemy models (44 tables), session, repository helpers, seed |
| `market_intel/` | normalisation, clustering, topics/drift, scoring |
| `prompt_ir/`, `prompt_lab/` | IR schema + compiler, niche templates, synthesis, mutation, optimiser |
| `evaluation/` | validators, stats, judge + calibration, robustness, suites, runner |
| `governance/` | policy, similarity/IP, injection gate, approval/publish/rollback |
| `publishing/` | deterministic packager, pricing/listings, storefront adapters |
| `experimentation/` | bandits, experiment statistics, service |
| `observability/` | structured logs, metrics, tracing |
| `connectors/` | upload, mock, official-API connectors behind one interface |
| `workers/` | Celery app + idempotent tasks, beat schedule |
| `infra/` | Dockerfile, compose (Postgres+pgvector, Redis, Prometheus, Grafana) |
| `demo/` | synthetic dataset, billing test suites, permitted corpus, end-to-end demo |

## What the demo proves

```
STEP 2  65 signals → 10 clusters → 10 scored opportunities
STEP 3  ranking by 90% bootstrap lower bound (not the noisiest headline number)
STEP 4  3 candidates → screened → finalist score 0.78 [min 0.70], 11/11 attacks resisted
STEP 5  publishing blocked: 409 publish_blocked (no approved approval record)
STEP 6  approval → signed deterministic pack + listing (mock storefront)
STEP 7  ~320 visitors: inconclusive → ~1000 visitors: +72.8% uplift, p=0.019 → shipped
STEP 8  cost ledger, budget utilisation, audit trail
```

Scores come from the deterministic simulator: they prove the harness, gates and
statistics work — not that a model is good. Set a provider key and re-run
`POST /versions/{id}/evaluate` for real evidence.

## Configuration

Copy `.env.example` to `.env`. Nothing is required. Safe defaults: human approval on,
auto-publish off, budgets enforced, mock storefront, eager Celery, SQLite. See
`.env.example` for Postgres/Redis, provider keys, connector credentials and the
`PF_SIGNING_KEY` used to sign packs.

## Status

Working end-to-end: ingestion, discovery, scoring, generation, optimisation, evaluation,
governance, approval, publishing, experiments, observability, migrations, demo and tests.
Next milestones: broader test coverage (API/pipeline/gate/experiment), the full design
document, and a hosted review UI.
