#!/usr/bin/env python
"""PromptFoundry end-to-end demo — runs with **zero API keys and no network access**.

What it proves
--------------
1. Demand signals (synthetic CSV + mocked official-API connector) → normalised signals.
2. Clustering + scoring produce ranked, evidence-backed opportunities.
3. The best opportunity with a matching test suite becomes a product version:
   Prompt-IR → multi-family compilation → staged evaluation (deterministic + calibrated
   judge + robustness) → governance gate.
4. Publishing is *blocked* until a human with the ``approver`` role approves.
5. Publishing packages a signed, deterministic pack and produces a storefront listing.
6. Online experimentation assigns traffic deterministically, measures the uplift with a
   confidence interval, and refuses to act when the sample is too small.
7. Everything above is in the audit log and the cost ledger.

Usage
-----
    python -m demo.run_demo          # or: make demo
"""

from __future__ import annotations

import sys
import time
from typing import Any

from fastapi.testclient import TestClient

from backend.main import app
from core.config import get_settings
from db.seed import DEMO_USERS, seed_all

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
RESET = "\033[0m"

STEP = 0


def header(text: str) -> None:
    global STEP
    STEP += 1
    print(f"\n{BOLD}{CYAN}{'=' * 78}\nSTEP {STEP}. {text}\n{'=' * 78}{RESET}")


def say(text: str, colour: str = "") -> None:
    print(f"{colour}{text}{RESET}")


def table(rows: list[dict[str, Any]], columns: list[str], max_width: int = 34) -> None:
    if not rows:
        say("  (none)", DIM)
        return
    widths = {
        column: min(max_width, max(len(column), *(len(str(row.get(column, ""))) for row in rows)))
        for column in columns
    }
    header_line = "  " + "  ".join(column.ljust(widths[column]) for column in columns)
    print(DIM + header_line + RESET)
    for row in rows:
        cells = []
        for column in columns:
            value = str(row.get(column, ""))
            cells.append(
                value[: widths[column] - 1] + "…"
                if len(value) > widths[column]
                else value.ljust(widths[column])
            )
        print("  " + "  ".join(cells))


class Api:
    """Tiny authenticated HTTP client over the ASGI app (no server, no ports)."""

    def __init__(self, client: TestClient) -> None:
        self.client = client
        self.token: str | None = None
        self.role = ""

    def login(self, email: str, password: str, role: str) -> dict[str, Any]:
        response = self.client.post("/auth/login", json={"email": email, "password": password})
        if response.status_code != 200:
            raise SystemExit(f"login failed for {email}: {response.status_code} {response.text}")
        payload = response.json()
        self.token = payload["token"]
        self.role = role
        return payload

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def get(self, path: str, **kwargs: Any) -> Any:
        response = self.client.get(path, headers=self.headers, **kwargs)
        return self._unpack("GET", path, response)

    def post(self, path: str, payload: Any = None, expect: int | None = None) -> Any:
        response = self.client.post(path, json=payload or {}, headers=self.headers)
        return self._unpack("POST", path, response, expect=expect)

    def _unpack(self, method: str, path: str, response: Any, expect: int | None = None) -> Any:
        if expect is not None and response.status_code == expect:
            return response.json() if response.content else {}
        if response.status_code >= 400:
            raise RuntimeError(f"{method} {path} → {response.status_code}: {response.text[:600]}")
        return response.json() if response.content else {}


def main() -> int:
    settings = get_settings()
    started = time.perf_counter()

    print(f"{BOLD}PromptFoundry demo{RESET}")
    say(f"env={settings.env}  db={settings.database_url}", DIM)
    say(
        f"llm keys: openai={'set' if settings.openai_api_key else 'none'} "
        f"anthropic={'set' if settings.anthropic_api_key else 'none'} "
        f"google={'set' if settings.google_api_key else 'none'} → "
        f"using the deterministic mock simulator",
        DIM,
    )
    say(
        f"human approval required={settings.require_human_approval}  "
        f"auto-publish={settings.auto_publish_enabled}  "
        f"weekly LLM budget=A${settings.budget_weekly_llm_micros / 1_000_000:.0f}",
        DIM,
    )

    header("Seed the workspace (idempotent)")
    seed = seed_all(run_discovery_pass=False)
    for note in seed["notes"]:
        say(f"  • {note}", DIM)
    for kind, count in seed["created"].items():
        say(f"  + {count} {kind}", GREEN)

    with TestClient(app) as client:
        owner = Api(client)
        owner.login(DEMO_USERS[0]["email"], DEMO_USERS[0]["password"], "owner (admin)")
        approver = Api(client)
        approver.login(DEMO_USERS[2]["email"], DEMO_USERS[2]["password"], "approver")
        say(
            f"  logged in as {owner.role} and {approver.role} "
            f"(separate identities: creation ≠ approval)",
            GREEN,
        )

        header("Ingest + discover demand (no scraping, no keys)")
        sources = owner.get("/sources")["items"]
        table(
            [
                {
                    "source": row["name"][:28],
                    "type": row["source_type"],
                    "permission": row["permission_basis"],
                    "signals": (row.get("stats") or {}).get("signals", 0),
                }
                for row in sources
            ],
            ["source", "type", "permission", "signals"],
        )
        discovery = owner.post("/pipeline/discover", {"max_signals": 500})
        say(
            f"  {discovery['signals_considered']} signals → "
            f"{discovery['clusters_found']} clusters → "
            f"{len(discovery['opportunities'])} scored opportunities",
            GREEN,
        )

        header("Rank opportunities (profit-first, uncertainty-aware)")
        ranking = owner.get("/opportunities?limit=6")["items"]
        table(
            [
                {
                    "score": f"{row['composite']:.3f}",
                    "lower90": f"{row['ci'][0]:.3f}",
                    "conf": f"{row['confidence']:.2f}",
                    "price": f"A${row['price_hypothesis_aud']:.0f}",
                    "title": row["title"],
                }
                for row in ranking
            ],
            ["score", "lower90", "conf", "price", "title"],
        )
        say(
            "  ranking uses the 90% bootstrap lower bound, so noisy 'big' ideas "
            "do not outrank well-evidenced ones",
            DIM,
        )

        # Prefer an opportunity that has an evaluation suite; otherwise take the best.
        suites = owner.get("/suites")["items"]
        niches_with_suites = {row["niche"] for row in suites if not row["is_holdout"]}
        target = next((row for row in ranking if row["niche"] in niches_with_suites), ranking[0])
        if target["niche"] not in niches_with_suites:
            say(
                f"  {YELLOW}no suite for '{target['niche']}' — the gate will fall back to the "
                f"closest suite; add one under demo/data/tests{RESET}"
            )

        header(f"Build + evaluate: {target['title']}")
        launched = owner.post(
            f"/opportunities/{target['id']}/launch",
            {"n_candidates": 3, "evaluate_holdout": True, "run_gate": True, "price_aud": 39},
        )
        version_id = launched["version_id"]
        say(f"  product {launched['product_id']}  version {launched.get('version', '1.0.0')}", DIM)
        for step in launched["steps"]:
            ok = step.get("ok", True) and step.get("status") not in {"failed", "blocked"}
            name = step.get("step") or step.get("name") or "step"
            extras = {
                key: value
                for key, value in step.items()
                if key not in {"step", "name", "ok", "status"}
            }
            rendered = ", ".join(f"{key}={value}" for key, value in extras.items())
            say(f"  {(GREEN + '✓') if ok else (RED + '✗')}{RESET} {name}: {rendered}")

        optimisation = launched.get("optimization") or {}
        if optimisation.get("candidates"):
            say("\n  candidate screening (cost-capped, staged):", BOLD)
            table(
                [
                    {
                        "variant": row["variation"],
                        "screen": row.get("screen_score"),
                        "offline": row.get("offline_score"),
                        "final": row.get("finalist_score"),
                        "outcome": row.get("eliminated_at") or "winner",
                    }
                    for row in optimisation["candidates"]
                ],
                ["variant", "screen", "offline", "final", "outcome"],
            )

        detail = owner.get(f"/versions/{version_id}")
        summary = detail.get("metrics") or {}
        evaluations = detail.get("evaluations") or []
        finalist = next(
            (row for row in evaluations if row.get("stage") == "finalist"),
            evaluations[0] if evaluations else {},
        )
        say(
            f"\n  staged scores — screen={summary.get('screen')} offline={summary.get('offline')} "
            f"finalist={summary.get('finalist')}",
            GREEN,
        )
        if finalist.get("run_id"):
            run_detail = owner.get(f"/evaluations/{finalist['run_id']}")
            run_summary = run_detail.get("summary") or {}
            say(
                f"  finalist run {finalist['run_id']}: status={run_detail['status']} "
                f"score={run_detail['score']} passed={run_summary.get('passed')} "
                f"min={run_summary.get('min_score')} "
                f"cost=A${run_summary.get('cost_aud', 0):.4f} "
                f"(shadow A${run_summary.get('shadow_cost_aud', 0):.4f} at reference prices)",
                GREEN,
            )
            for metric in (run_detail.get("metrics") or [])[:6]:
                ci = metric.get("ci") or [None, None]
                ci_text = (
                    f" [{ci[0]:.3f}, {ci[1]:.3f}]"
                    if ci[0] is not None and ci[1] is not None
                    else ""
                )
                say(
                    f"    - {metric['metric']}: {metric['value']:.3f}{ci_text} (n={metric['n']})",
                    DIM,
                )
            robustness = run_summary.get("robustness") or {}
            if robustness:
                say(
                    f"    robustness: {robustness.get('passed', 0)}/{robustness.get('total', 0)} "
                    f"attacks resisted (pass_rate={robustness.get('pass_rate')})",
                    DIM,
                )

        gate = launched.get("gate") or {}
        decision = gate.get("decision", "unknown")
        colour = GREEN if decision in {"pass", "approved", "auto_approved"} else YELLOW
        say(
            f"  governance gate: {colour}{decision}{RESET} (worst-of policy, similarity, eval, injection, risk)"
        )
        for blocker in gate.get("blockers", []):
            say(f"    ✗ {blocker}", RED)
        for warning in gate.get("warnings", []):
            say(f"    ! {warning}", YELLOW)

        header("Human approval (publishing is blocked by default)")
        blocked = client.post(
            f"/versions/{version_id}/publish", json={"price_aud": 39}, headers=approver.headers
        )
        say(
            f"  publish before approval → {blocked.status_code} "
            f"{blocked.json().get('error', {}).get('code', '')} "
            f"({blocked.json().get('error', {}).get('message', '')[:90]})",
            GREEN,
        )
        approvals = approver.get("/approvals")["items"]
        pending = [
            row
            for row in approvals
            if row["subject_id"] == version_id and row["status"] == "pending"
        ]
        if not pending:
            say(f"  {YELLOW}no pending approval found; publishing may already be complete{RESET}")
        else:
            approval_id = pending[0]["approval_id"]
            decided = approver.post(
                f"/approvals/{approval_id}/approve",
                {"reason": "demo review: checks pass, claims are narrow and truthful"},
                expect=200,
            )
            say(f"  approver decision: {decided['status']} (audited, actor-attributed)", GREEN)

        header("Publish: signed pack + storefront listing (mock adapter)")
        published = approver.post(f"/versions/{version_id}/publish", {"price_aud": 39}, expect=200)
        say(
            f"  status={published['status']} storefront={published.get('storefront')} "
            f"external_id={published.get('external_id')} url={published.get('url')}"
        )
        say(
            f"  pack uri={published.get('pack_uri')} signature={published.get('pack_signature')}",
            DIM,
        )
        listing = published.get("listing") or {}
        say(f"  listing: {BOLD}{listing.get('title')}{RESET} — A${listing.get('price_aud')}")
        for bullet in (listing.get("bullets") or [])[:3]:
            say(f"    • {bullet}", DIM)

        header("Online experiment: A/B with guardrails and honest statistics")
        variants = owner.get(f"/versions/{version_id}")["variants"]
        control_variant = next((v for v in variants if v["is_primary"]), variants[0])
        treatment_variant = next(
            (v for v in variants if v["id"] != control_variant["id"]), control_variant
        )
        experiment = owner.post(
            "/experiments",
            {
                "product_id": launched["product_id"],
                "name": f"Model-family test for {target['title'][:40]}",
                "hypothesis": "The Anthropic-tuned compilation converts better than the generic one",
                "kind": "ab_test",
                "traffic_fraction": 1.0,
                "min_sample": 120,
                "arms": [
                    {
                        "name": "control-generic",
                        "version_id": version_id,
                        "variant_id": control_variant["id"],
                        "weight": 0.5,
                        "is_control": True,
                    },
                    {
                        "name": "treatment-anthropic",
                        "version_id": version_id,
                        "variant_id": treatment_variant["id"],
                        "weight": 0.5,
                        "is_control": False,
                    },
                ],
            },
        )
        experiment_id = experiment["experiment_id"]
        say(f"  experiment {experiment_id} → arms {[a['name'] for a in experiment['arms']]}", DIM)

        # Deterministic conversion pattern (5% control, 9% treatment) so the demo tells
        # the same honest story on every machine instead of depending on a random draw.
        exposures = {"control-generic": 0, "treatment-anthropic": 0}
        purchases = {"control-generic": 0, "treatment-anthropic": 0}
        next_visitor = 0

        def simulate(visitors: int) -> None:
            nonlocal next_visitor
            for _ in range(visitors):
                assignment = owner.post(
                    f"/experiments/{experiment_id}/assign",
                    {"subject_key": f"visitor-{next_visitor}"},
                )
                next_visitor += 1
                if not assignment.get("assigned"):
                    continue
                arm = assignment["arm_name"]
                index_in_arm = exposures[arm]
                exposures[arm] += 1
                threshold = 5 if arm == "control-generic" else 9
                if (index_in_arm * 7919) % 100 < threshold:
                    purchases[arm] += 1
                    owner.post(
                        f"/experiments/{experiment_id}/conversions",
                        {
                            "exposure_id": assignment["exposure_id"],
                            "event_type": "purchase",
                            "value_aud": 39.0,
                            "order_ref": f"demo-order-{next_visitor}",
                        },
                    )

        def report(analysis: dict[str, Any], label: str) -> None:
            table(
                [
                    {
                        "arm": arm["name"],
                        "visitors": arm["exposures"],
                        "purchases": arm["conversions"],
                        "rate": f"{(arm.get('conversion_rate') or {}).get('estimate', 0):.1%}",
                        "95% CI": (
                            f"{(arm.get('conversion_rate') or {}).get('low', 0):.1%}"
                            f"–{(arm.get('conversion_rate') or {}).get('high', 0):.1%}"
                        ),
                        "revenue": f"A${arm.get('revenue_aud', 0):.0f}",
                    }
                    for arm in analysis.get("arms") or []
                ],
                ["arm", "visitors", "purchases", "rate", "95% CI", "revenue"],
            )
            colour = GREEN if analysis.get("recommendation") == "ship_treatment" else YELLOW
            say(
                f"  {label}: uplift={analysis.get('uplift')} p={analysis.get('p_value')} "
                f"sample_ok={analysis.get('sample_ok')} → "
                f"{colour}{BOLD}{analysis.get('recommendation')}{RESET} — {analysis.get('reason')}",
            )

        simulate(320)
        first = owner.get(f"/experiments/{experiment_id}/analysis")
        report(first, "checkpoint @ ~320 visitors")
        say(
            "  (an early 'not significant' is the honest answer — no shipping on vibes)",
            DIM,
        )

        simulate(700)
        second = owner.get(f"/experiments/{experiment_id}/analysis")
        report(second, "checkpoint @ ~1000 visitors")
        promote = second.get("recommendation") == "ship_treatment"
        decided = owner.post(
            f"/experiments/{experiment_id}/decide", {"promote_winning_arm": promote}
        )
        say(f"  decision applied: {decided['action']}", GREEN if promote else DIM)

        header("Cost, budget and audit trail")
        budget = owner.get("/ops/budget")
        say(
            f"  LLM+infra spend today A${budget['daily_spent_aud']:.4f} / A${budget['daily_limit_aud']:.2f}, "
            f"this week A${budget['weekly_spent_aud']:.4f} / A${budget['weekly_limit_aud']:.2f} "
            f"({budget['utilisation']:.1%} of the binding cap)",
            GREEN,
        )
        audit = owner.get("/audit?limit=8")["items"]
        table(
            [
                {
                    "action": row["action"],
                    "actor": row["actor"]["id"][:14],
                    "decision": row["decision"],
                }
                for row in audit
            ],
            ["action", "actor", "decision"],
        )
        say(
            f"\n{BOLD}Demo complete in {time.perf_counter() - started:.1f}s.{RESET} "
            f"Artifacts are in {settings.artifacts_dir}.",
        )
        say(
            "Everything you just saw ran with zero API keys, zero scraping and "
            "human-approved publishing.",
            DIM,
        )
        say(
            "Scores come from the deterministic mock simulator: they prove the harness, "
            "gates and statistics work, not that a model is good. Set an API key and "
            "re-run `POST /versions/{id}/evaluate` for real evidence.",
            DIM,
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:  # pragma: no cover
        sys.exit(130)
