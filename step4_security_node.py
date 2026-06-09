"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 4 — ADD THE SECURITY SCANNER NODE
 "A specialist node. One job: OWASP Top 10."
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 What's new:
   ✦ security_scan node added AFTER parse_code
   ✦ Structured JSON output — every finding has severity + line
   ✦ Graph is now 2 nodes: parse → security → END

 Run: python step4_security_node.py

 SHOW THE AUDIENCE:
   - Each finding has severity, line_number, owasp ref, code_fix
   - Run JUST this node: it returns immediately if no critical issues
   - The node is completely independent — swap Claude for GPT-4 here only
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json, time, os
import anthropic
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()


class ReviewState(TypedDict):
    code: str;         filename: str
    language: str;     line_count: int;    functions: List[str]
    security_issues: List[dict];           bug_issues: List[dict]
    quality_issues: List[dict];            all_issues: List[dict]
    risk_score: int;   risk_label: str;    final_report: str
    needs_human_review: bool;              human_approved: Optional[bool]
    token_usage: dict; processing_steps: List[str]; elapsed_ms: dict
    error: Optional[str];                  retry_count: int


# ── NODE 1 (same as step 3) ─────────────────────────────────────────
def parse_code(state: ReviewState) -> dict:
    print(f"  [1/2] parse_code ...")
    t0 = time.time()
    response = client.messages.create(
        model="claude-opus-4-5", max_tokens=512,
        system="Return ONLY valid JSON, no markdown.",
        messages=[{"role": "user", "content":
            f'Return JSON: {{"language":"<lang>","line_count":<n>,"functions":["..."]}}\n\nCODE:\n{state["code"]}'}]
    )
    data = json.loads(response.content[0].text)
    elapsed = (time.time() - t0) * 1000
    tokens = response.usage.input_tokens + response.usage.output_tokens
    return {
        "language": data.get("language", "unknown"),
        "line_count": data.get("line_count", 0),
        "functions": data.get("functions", []),
        "processing_steps": state.get("processing_steps", []) + ["parse_code"],
        "token_usage": {**state.get("token_usage", {}), "parse_code": tokens},
        "elapsed_ms": {**state.get("elapsed_ms", {}), "parse_code": round(elapsed, 1)},
    }


# ── NODE 2: security_scan ───────────────────────────────────────────
# NEW: A security specialist. Focused only on vulnerabilities.
# Returns structured findings — NOT raw text.

def security_scan(state: ReviewState) -> dict:
    print(f"  [2/2] security_scan ({state['language']}) ...")
    t0 = time.time()

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2048,
        system="You are a senior application security engineer. Return ONLY valid JSON, no markdown.",
        messages=[{"role": "user", "content": f"""Find ALL security vulnerabilities in this {state['language']} code.

Return JSON:
{{
  "issues": [
    {{
      "severity": "critical|high|medium|low",
      "line_number": <number or null>,
      "title": "<short title>",
      "description": "<what it is and why dangerous>",
      "suggestion": "<how to fix>",
      "code_fix": "<fixed code or null>",
      "owasp": "<e.g. A03:2021 Injection>"
    }}
  ],
  "has_critical": true|false,
  "summary": "<1 sentence>"
}}

CODE:
{state['code']}"""}]
    )

    data = json.loads(response.content[0].text)
    elapsed = (time.time() - t0) * 1000
    tokens = response.usage.input_tokens + response.usage.output_tokens

    issues = data.get("issues", [])
    print(f"         → Found {len(issues)} security issues "
          f"({'⚠️  CRITICAL' if data.get('has_critical') else 'no critical'})")

    return {
        "security_issues": issues,
        "processing_steps": state.get("processing_steps", []) + ["security_scan"],
        "token_usage": {**state.get("token_usage", {}), "security_scan": tokens},
        "elapsed_ms": {**state.get("elapsed_ms", {}), "security_scan": round(elapsed, 1)},
    }


# ── Graph: parse → security → END ──────────────────────────────────
def build_graph():
    workflow = StateGraph(ReviewState)
    workflow.add_node("parse_code",    parse_code)
    workflow.add_node("security_scan", security_scan)  # ← NEW
    workflow.set_entry_point("parse_code")
    workflow.add_edge("parse_code",    "security_scan") # ← NEW edge
    workflow.add_edge("security_scan", END)
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

    print("Running 2-node pipeline...\n")
    result = app.invoke(initial_state, config={"configurable": {"thread_id": "demo"}})

    print(f"\n{'='*60}")
    print(f"SECURITY FINDINGS ({len(result['security_issues'])} total)")
    print(f"{'='*60}")
    for i, issue in enumerate(result["security_issues"], 1):
        sev = issue.get("severity", "?").upper()
        line = f"line {issue['line_number']}" if issue.get("line_number") else "no line"
        print(f"\n  [{i}] [{sev}] {issue.get('title')} ({line})")
        print(f"       {issue.get('description', '')[:80]}...")
        if issue.get("owasp"):
            print(f"       OWASP: {issue['owasp']}")
        if issue.get("code_fix"):
            print(f"       FIX: {issue['code_fix'][:60]}...")

    total_tokens = sum(result["token_usage"].values())
    total_ms = sum(result["elapsed_ms"].values())
    print(f"\n⚡ {round(total_ms/1000,1)}s  |  {total_tokens:,} tokens used")
    print(f"   Steps run: {' → '.join(result['processing_steps'])}")
