"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 5 — THE FULL PIPELINE + HUMAN-IN-THE-LOOP
 "All 5 nodes. Risk score. HITL gate."
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 What's new:
   ✦ bug_detection node added
   ✦ aggregate_issues node — calculates risk score 0-100
   ✦ should_escalate() — conditional edge (HITL trigger)
   ✦ human_review_gate — the graph PAUSES here for a human
   ✦ generate_report — final structured output

 Graph:
   parse → security → bugs → aggregate → [HITL?] → report

 Run: python step5_full_pipeline.py

 THE DEMO MOMENT:
   When the agent hits 3+ critical issues it will PAUSE and ask
   for your approval before continuing. Type 'y' to approve.
   This is Human-in-the-Loop in action.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json, time, os
import anthropic
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()
from agent_prism import make_client, report_review, MODEL
client = make_client()


class ReviewState(TypedDict):
    code: str;         filename: str
    language: str;     line_count: int;    functions: List[str]
    security_issues: List[dict];           bug_issues: List[dict]
    quality_issues: List[dict];            all_issues: List[dict]
    risk_score: int;   risk_label: str;    final_report: str
    needs_human_review: bool;              human_approved: Optional[bool]
    token_usage: dict; processing_steps: List[str]; elapsed_ms: dict
    error: Optional[str];                  retry_count: int


def _call_claude(system: str, user: str) -> tuple[dict, int]:
    """Helper: call Claude, parse JSON, return (data, tokens)."""
    for attempt in range(1, 4):
        try:
            r = client.messages.create(
                model=MODEL, max_tokens=2048,
                system=system, messages=[{"role": "user", "content": user}]
            )
            text = r.content[0].text.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:-1])
            return json.loads(text), r.usage.input_tokens + r.usage.output_tokens
        except json.JSONDecodeError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


# ── NODE 1: parse_code ──────────────────────────────────────────────
def parse_code(state: ReviewState) -> dict:
    print("  [1/5] parse_code ...")
    t0 = time.time()
    data, tokens = _call_claude(
        "Return ONLY valid JSON.",
        f'Return {{"language":"<lang>","line_count":<n>,"functions":["..."]}}\nCODE:\n{state["code"]}'
    )
    return {
        "language": data.get("language", "unknown"),
        "line_count": data.get("line_count", 0),
        "functions": data.get("functions", []),
        "processing_steps": state.get("processing_steps", []) + ["parse_code"],
        "token_usage": {**state.get("token_usage", {}), "parse_code": tokens},
        "elapsed_ms": {**state.get("elapsed_ms", {}), "parse_code": round((time.time()-t0)*1000, 1)},
    }


# ── NODE 2: security_scan ───────────────────────────────────────────
def security_scan(state: ReviewState) -> dict:
    print("  [2/5] security_scan ...")
    t0 = time.time()
    data, tokens = _call_claude(
        "Senior security engineer. Return ONLY valid JSON.",
        f"""Find ALL security vulnerabilities in this {state['language']} code.
Return JSON: {{"issues":[{{"severity":"critical|high|medium|low","line_number":<n>,"title":"","description":"","suggestion":"","code_fix":null,"owasp":""}}],"has_critical":false}}
CODE:\n{state['code']}"""
    )
    issues = data.get("issues", [])
    print(f"         → {len(issues)} security issues")
    return {
        "security_issues": issues,
        "processing_steps": state.get("processing_steps", []) + ["security_scan"],
        "token_usage": {**state.get("token_usage", {}), "security_scan": tokens},
        "elapsed_ms": {**state.get("elapsed_ms", {}), "security_scan": round((time.time()-t0)*1000, 1)},
    }


# ── NODE 3: bug_detection ───────────────────────────────────────────
def bug_detection(state: ReviewState) -> dict:
    print("  [3/5] bug_detection ...")
    t0 = time.time()
    data, tokens = _call_claude(
        "Senior software engineer. Return ONLY valid JSON.",
        f"""Find runtime bugs in this {state['language']} code.
Return JSON: {{"issues":[{{"severity":"critical|high|medium|low","line_number":<n>,"title":"","description":"","suggestion":"","code_fix":null}}]}}
CODE:\n{state['code']}"""
    )
    issues = data.get("issues", [])
    print(f"         → {len(issues)} bugs")
    return {
        "bug_issues": issues,
        "processing_steps": state.get("processing_steps", []) + ["bug_detection"],
        "token_usage": {**state.get("token_usage", {}), "bug_detection": tokens},
        "elapsed_ms": {**state.get("elapsed_ms", {}), "bug_detection": round((time.time()-t0)*1000, 1)},
    }


# ── NODE 4: aggregate_issues ────────────────────────────────────────
def aggregate_issues(state: ReviewState) -> dict:
    print("  [4/5] aggregate_issues ...")
    all_issues = (
        state.get("security_issues", []) +
        state.get("bug_issues", []) +
        state.get("quality_issues", [])
    )
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for issue in all_issues:
        sev = issue.get("severity", "low").lower()
        counts[sev] = counts.get(sev, 0) + 1

    # Risk formula: critical×25 + high×10 + medium×5 + low×1, capped at 100
    score = min(100,
        counts["critical"] * 25 + counts["high"] * 10 +
        counts["medium"] * 5  + counts["low"] * 1
    )
    label = ("CRITICAL" if score >= 75 else "HIGH" if score >= 50
             else "MEDIUM" if score >= 25 else "LOW")
    needs_human = counts["critical"] > 0

    print(f"         → Risk {score}/100 ({label}) | {len(all_issues)} total issues")
    if needs_human:
        print(f"         → ⚠️  CRITICAL issues found — will escalate to human review")

    return {
        "all_issues": all_issues,
        "risk_score": score,
        "risk_label": label,
        "needs_human_review": needs_human,
        "processing_steps": state.get("processing_steps", []) + ["aggregate_issues"],
    }


# ── CONDITIONAL EDGE: should_escalate? ─────────────────────────────
# This function determines which node runs next.
# "escalate" → human_review_gate
# "report"   → generate_report  (skips human gate)

def should_escalate(state: ReviewState) -> str:
    return "escalate" if state.get("needs_human_review") else "report"


# ── NODE 5a: human_review_gate (HITL) ──────────────────────────────
def human_review_gate(state: ReviewState) -> dict:
    print("\n" + "🚨" * 30)
    print("HUMAN REVIEW REQUIRED")
    print("🚨" * 30)
    print(f"  File      : {state['filename']}")
    print(f"  Risk Score: {state['risk_score']}/100 — {state['risk_label']}")
    print(f"  Critical  : {sum(1 for i in state['all_issues'] if i.get('severity')=='critical')}")
    print("\n  Critical findings:")
    for issue in state.get("all_issues", []):
        if issue.get("severity") == "critical":
            print(f"    ❌ {issue.get('title')} — {issue.get('description','')[:60]}...")

    answer = input("\n  Approve this review? [y/n]: ").strip().lower()
    return {
        "human_approved": answer == "y",
        "processing_steps": state.get("processing_steps", []) + ["human_review_gate"],
    }


# ── NODE 5b: generate_report ────────────────────────────────────────
def generate_report(state: ReviewState) -> dict:
    print("  [5/5] generate_report ...")
    all_issues = state.get("all_issues", [])
    counts = {}
    for i in all_issues:
        s = i.get("severity", "low")
        counts[s] = counts.get(s, 0) + 1

    report = f"""# Code Review Report: {state['filename']}

**Risk Score:** {state['risk_score']}/100 — {state['risk_label']}
**Issues:** {len(all_issues)} total
  - Critical: {counts.get('critical', 0)}
  - High:     {counts.get('high', 0)}
  - Medium:   {counts.get('medium', 0)}
  - Low:      {counts.get('low', 0)}

## All Findings (sorted by severity)
"""
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    sorted_issues = sorted(all_issues, key=lambda x: severity_order.get(x.get("severity","low"), 4))
    for i, issue in enumerate(sorted_issues, 1):
        line = f" (line {issue['line_number']})" if issue.get("line_number") else ""
        report += f"\n### [{i}] [{issue.get('severity','').upper()}] {issue.get('title','')}{line}\n"
        report += f"{issue.get('description','')}\n"
        report += f"**Fix:** {issue.get('suggestion','')}\n"
        if issue.get("code_fix"):
            report += f"```\n{issue['code_fix']}\n```\n"

    return {
        "final_report": report,
        "processing_steps": state.get("processing_steps", []) + ["generate_report"],
    }


# ── Build the full graph ─────────────────────────────────────────────
def build_graph():
    workflow = StateGraph(ReviewState)
    workflow.add_node("parse_code",        parse_code)
    workflow.add_node("security_scan",     security_scan)
    workflow.add_node("bug_detection",     bug_detection)
    workflow.add_node("aggregate_issues",  aggregate_issues)
    workflow.add_node("human_review_gate", human_review_gate)
    workflow.add_node("generate_report",   generate_report)

    workflow.set_entry_point("parse_code")
    workflow.add_edge("parse_code",       "security_scan")
    workflow.add_edge("security_scan",    "bug_detection")
    workflow.add_edge("bug_detection",    "aggregate_issues")

    # THE KEY MOMENT: conditional routing
    workflow.add_conditional_edges(
        "aggregate_issues",
        should_escalate,
        {
            "escalate": "human_review_gate",
            "report":   "generate_report"
        }
    )
    workflow.add_edge("human_review_gate", "generate_report")
    workflow.add_edge("generate_report",   END)
    return workflow.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    app = build_graph()
    initial_state: ReviewState = {
        "code": open("samples/buggy_code.py").read(), "filename": "buggy_code.py",
        "language": "", "line_count": 0, "functions": [],
        "security_issues": [], "bug_issues": [], "quality_issues": [],
        "all_issues": [], "risk_score": 0, "risk_label": "", "final_report": "",
        "needs_human_review": False, "human_approved": None,
        "token_usage": {}, "processing_steps": [], "elapsed_ms": {},
        "error": None, "retry_count": 0
    }

    print("Running FULL 5-node LangGraph pipeline...\n")
    result = app.invoke(initial_state, config={"configurable": {"thread_id": "demo"}})

    print("\n" + "=" * 60)
    print(result["final_report"])
    print("=" * 60)

    total_tokens = sum(result["token_usage"].values())
    total_ms = sum(result["elapsed_ms"].values())
    print(f"\n⚡ {round(total_ms/1000,1)}s total  |  {total_tokens:,} tokens")
    print(f"   Steps: {' → '.join(result['processing_steps'])}")
    if result.get("human_approved") is not None:
        status = "✅ Approved" if result["human_approved"] else "❌ Rejected"
        print(f"   Human review: {status}")

    report_review(result, "buggy_code.py")
