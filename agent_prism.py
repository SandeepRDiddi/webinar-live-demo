"""
Agent Prism integration helper.

Set these env vars to route all Claude calls through Agent Prism and
send a review summary card to the dashboard after each review:

  AGENT_PRISM_URL=https://agent-prism.onrender.com
  AGENT_PRISM_API_KEY=acp_your_tenant_key_here

If either var is missing, the agent runs standalone (direct Anthropic).
"""

import os
import anthropic

PRISM_URL = os.getenv("AGENT_PRISM_URL", "").rstrip("/")
PRISM_KEY = os.getenv("AGENT_PRISM_API_KEY", "")
MODEL = os.getenv("CODE_REVIEW_MODEL", "claude-sonnet-4-6")


def make_client() -> anthropic.Anthropic:
    """Return Anthropic client — routed through Agent Prism when configured."""
    if PRISM_URL and PRISM_KEY:
        print(f"  [Agent Prism] routing via {PRISM_URL}")
        return anthropic.Anthropic(api_key=PRISM_KEY, base_url=PRISM_URL)
    return anthropic.Anthropic()


def report_review(state: dict, filename: str = "code.py") -> None:
    """
    POST a review summary to Agent Prism /api/ingest (fire-and-forget).
    Shows up in the dashboard as one run card with risk score + issue counts.
    No-op when AGENT_PRISM_URL / AGENT_PRISM_API_KEY are not set.
    """
    if not PRISM_URL or not PRISM_KEY:
        return

    import requests

    total_tokens = sum(state.get("token_usage", {}).values())
    total_ms = int(sum(state.get("elapsed_ms", {}).values()))
    counts: dict = {}
    for issue in state.get("all_issues", []):
        s = issue.get("severity", "low")
        counts[s] = counts.get(s, 0) + 1

    # Rough cost estimate: sonnet ~$3/M input + $15/M output — use $6/M blended
    cost_usd = round(total_tokens * 0.000006, 6)

    approved = state.get("human_approved")
    if approved is False:
        status = "failed"
    elif state.get("needs_human_review") and approved is None:
        status = "flagged"
    else:
        status = "success"

    payload = {
        "agentName": "Code Review Agent",
        "workflow": f"review:{filename}",
        "model": MODEL,
        "provider": "anthropic",
        "totalTokens": total_tokens,
        "costUsd": cost_usd,
        "latencyMs": total_ms,
        "status": status,
        "successRate": 100 if status == "success" else 0,
        "metadata": {
            "risk_score": state.get("risk_score", 0),
            "risk_label": state.get("risk_label", ""),
            "issues_critical": counts.get("critical", 0),
            "issues_high": counts.get("high", 0),
            "issues_medium": counts.get("medium", 0),
            "issues_low": counts.get("low", 0),
            "total_issues": len(state.get("all_issues", [])),
            "human_approved": approved,
            "nodes_executed": state.get("processing_steps", []),
        },
    }

    try:
        resp = requests.post(
            f"{PRISM_URL}/api/ingest",
            headers={"X-API-Key": PRISM_KEY},
            json=payload,
            timeout=5,
        )
        if resp.status_code == 200:
            print(f"  [Agent Prism] review logged — risk {state.get('risk_score',0)}/100, ${cost_usd:.6f}")
        else:
            print(f"  [Agent Prism] ingest warn: HTTP {resp.status_code}")
    except Exception as e:
        print(f"  [Agent Prism] ingest skipped: {e}")
