"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 6 — ADD OBSERVABILITY
 "If you can't see it, you can't own it."
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 What's new:
   ✦ Structured JSON logs emitted at every node entry/exit
   ✦ Per-node timing visible in real time
   ✦ Token cost tracked per node
   ✦ human_review_required event logged with full context

 Run: python step6_observability.py

 DEMO TALKING POINT:
   The logs emit as each node runs.
   Point to the screen: "This is what Datadog / CloudWatch
   sees in production. You can build alerts on 'event':'human_review_required'."

 Diff from step5: ONLY the _call_node() wrapper is new.
   Every other line is identical. Observability is orthogonal.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json, time, os, logging, sys
import anthropic
from typing import TypedDict, List, Optional
from datetime import datetime, timezone
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

# ── Structured logger ───────────────────────────────────────────────
logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)
log = logging.getLogger("agent")

def emit(level: str, event: str, **kwargs):
    record = {
        "ts":      datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "service": "code-review-agent",
        "level":   level,
        "event":   event,
        **kwargs
    }
    log.info(json.dumps(record))


class ReviewState(TypedDict):
    code: str;         filename: str
    language: str;     line_count: int;    functions: List[str]
    security_issues: List[dict];           bug_issues: List[dict]
    quality_issues: List[dict];            all_issues: List[dict]
    risk_score: int;   risk_label: str;    final_report: str
    needs_human_review: bool;              human_approved: Optional[bool]
    token_usage: dict; processing_steps: List[str]; elapsed_ms: dict
    error: Optional[str];                  retry_count: int


def _call_claude(system: str, user: str, node_name: str) -> tuple[dict, int, float]:
    """Call Claude with retry. Returns (data, tokens, elapsed_ms)."""
    for attempt in range(1, 4):
        try:
            t0 = time.time()
            r = client.messages.create(
                model="claude-opus-4-5", max_tokens=2048,
                system=system, messages=[{"role": "user", "content": user}]
            )
            text = r.content[0].text.strip()
            if text.startswith("```"): text = "\n".join(text.split("\n")[1:-1])
            data = json.loads(text)
            elapsed = (time.time() - t0) * 1000
            tokens = r.usage.input_tokens + r.usage.output_tokens
            return data, tokens, round(elapsed, 1)
        except (json.JSONDecodeError, anthropic.APIError) as e:
            if attempt == 3: raise
            emit("WARN", "node_retry", node=node_name, attempt=attempt, reason=str(e))
            time.sleep(2 ** attempt)


# ── Nodes (identical logic, just with emit() calls added) ───────────

def parse_code(state: ReviewState) -> dict:
    emit("INFO", "node_started", node="parse_code", file=state["filename"])
    data, tokens, ms = _call_claude(
        "Return ONLY valid JSON.",
        f'Return {{"language":"<lang>","line_count":<n>,"functions":["..."]}}\nCODE:\n{state["code"]}',
        "parse_code"
    )
    emit("INFO", "node_completed", node="parse_code", elapsed_ms=ms, tokens=tokens)
    return {
        "language": data.get("language","unknown"), "line_count": data.get("line_count",0),
        "functions": data.get("functions",[]),
        "processing_steps": state.get("processing_steps",[]) + ["parse_code"],
        "token_usage": {**state.get("token_usage",{}), "parse_code": tokens},
        "elapsed_ms": {**state.get("elapsed_ms",{}), "parse_code": ms},
    }

def security_scan(state: ReviewState) -> dict:
    emit("INFO", "node_started", node="security_scan", file=state["filename"])
    data, tokens, ms = _call_claude(
        "Senior security engineer. Return ONLY valid JSON.",
        f"""Find ALL security vulnerabilities in this {state['language']} code.
Return JSON: {{"issues":[{{"severity":"critical|high|medium|low","line_number":<n>,"title":"","description":"","suggestion":"","code_fix":null,"owasp":""}}],"has_critical":false}}
CODE:\n{state['code']}""", "security_scan"
    )
    issues = data.get("issues", [])
    emit("INFO", "node_completed", node="security_scan", elapsed_ms=ms, tokens=tokens, issues_found=len(issues))
    return {
        "security_issues": issues,
        "processing_steps": state.get("processing_steps",[]) + ["security_scan"],
        "token_usage": {**state.get("token_usage",{}), "security_scan": tokens},
        "elapsed_ms": {**state.get("elapsed_ms",{}), "security_scan": ms},
    }

def bug_detection(state: ReviewState) -> dict:
    emit("INFO", "node_started", node="bug_detection", file=state["filename"])
    data, tokens, ms = _call_claude(
        "Senior  engineer. Return ONLY valid JSON.",
        f"""Find runtime bugs in this {state['language']} code.
Return JSON: {{"issues":[{{"severity":"critical|high|medium|low","line_number":<n>,"title":"","description":"","suggestion":"","code_fix":null}}]}}
CODE:\n{state['code']}""", "bug_detection"
    )
    issues = data.get("issues", [])
    emit("INFO", "node_completed", node="bug_detection", elapsed_ms=ms, tokens=tokens, issues_found=len(issues))
    return {
        "bug_issues": issues,
        "processing_steps": state.get("processing_steps",[]) + ["bug_detection"],
        "token_usage": {**state.get("token_usage",{}), "bug_detection": tokens},
        "elapsed_ms": {**state.get("elapsed_ms",{}), "bug_detection": ms},
    }

def aggregate_issues(state: ReviewState) -> dict:
    emit("INFO", "node_started", node="aggregate_issues", file=state["filename"])
    all_issues = state.get("security_issues",[]) + state.get("bug_issues",[]) + state.get("quality_issues",[])
    counts = {"critical":0,"high":0,"medium":0,"low":0}
    for i in all_issues:
        s = i.get("severity","low").lower()
        counts[s] = counts.get(s,0) + 1
    score = min(100, counts["critical"]*25 + counts["high"]*10 + counts["medium"]*5 + counts["low"])
    label = "CRITICAL" if score>=75 else "HIGH" if score>=50 else "MEDIUM" if score>=25 else "LOW"
    needs_human = counts["critical"] > 0
    if needs_human:
        emit("WARN", "human_review_required",
             file=state["filename"], risk_score=score, critical_issues=counts["critical"])
    emit("INFO", "node_completed", node="aggregate_issues", risk_score=score, label=label, total_issues=len(all_issues))
    return {
        "all_issues": all_issues, "risk_score": score, "risk_label": label,
        "needs_human_review": needs_human,
        "processing_steps": state.get("processing_steps",[]) + ["aggregate_issues"],
    }

def should_escalate(state: ReviewState) -> str:
    return "escalate" if state.get("needs_human_review") else "report"

def human_review_gate(state: ReviewState) -> dict:
    emit("WARN", "hitl_paused", file=state["filename"], risk_score=state["risk_score"])
    print("\n" + "="*50)
    print(f"🚨  HUMAN REVIEW — {state['filename']}  ({state['risk_score']}/100 {state['risk_label']})")
    print("="*50)
    for issue in state.get("all_issues",[]):
        if issue.get("severity") == "critical":
            print(f"  ❌ [{issue.get('severity','').upper()}] {issue.get('title','')}")
    answer = input("\n  Approve? [y/n]: ").strip().lower()
    approved = answer == "y"
    emit("INFO", "hitl_decision", file=state["filename"], approved=approved)
    return {
        "human_approved": approved,
        "processing_steps": state.get("processing_steps",[]) + ["human_review_gate"],
    }

def generate_report(state: ReviewState) -> dict:
    emit("INFO", "node_started", node="generate_report", file=state["filename"])
    all_issues = state.get("all_issues", [])
    counts = {}
    for i in all_issues:
        s = i.get("severity","low"); counts[s] = counts.get(s,0)+1
    severity_order = {"critical":0,"high":1,"medium":2,"low":3,"info":4}
    sorted_issues = sorted(all_issues, key=lambda x: severity_order.get(x.get("severity","low"),4))
    report = (
        f"# Review: {state['filename']}\n"
        f"Risk: {state['risk_score']}/100 — {state['risk_label']}\n"
        f"Issues: {len(all_issues)} ({counts.get('critical',0)} critical, {counts.get('high',0)} high, "
        f"{counts.get('medium',0)} medium, {counts.get('low',0)} low)\n\n"
    )
    for i, issue in enumerate(sorted_issues, 1):
        ln = f" (line {issue['line_number']})" if issue.get("line_number") else ""
        report += f"[{i}] [{issue.get('severity','').upper()}] {issue.get('title','')}{ln}\n"
        report += f"     {issue.get('description','')[:80]}\n"
        report += f"     FIX: {issue.get('suggestion','')[:80]}\n\n"
    total_tokens = sum(state.get("token_usage",{}).values())
    total_ms = sum(state.get("elapsed_ms",{}).values())
    emit("INFO", "review_complete", file=state["filename"],
         risk_score=state["risk_score"], total_issues=len(all_issues),
         total_ms=round(total_ms,1), total_tokens=total_tokens)
    return {
        "final_report": report,
        "processing_steps": state.get("processing_steps",[]) + ["generate_report"],
    }


def build_graph():
    wf = StateGraph(ReviewState)
    for name, fn in [("parse_code",parse_code),("security_scan",security_scan),
                     ("bug_detection",bug_detection),("aggregate_issues",aggregate_issues),
                     ("human_review_gate",human_review_gate),("generate_report",generate_report)]:
        wf.add_node(name, fn)
    wf.set_entry_point("parse_code")
    wf.add_edge("parse_code","security_scan")
    wf.add_edge("security_scan","bug_detection")
    wf.add_edge("bug_detection","aggregate_issues")
    wf.add_conditional_edges("aggregate_issues", should_escalate,
                             {"escalate":"human_review_gate","report":"generate_report"})
    wf.add_edge("human_review_gate","generate_report")
    wf.add_edge("generate_report", END)
    return wf.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    app = build_graph()
    state: ReviewState = {
        "code": open("samples/buggy_code.py").read(), "filename": "buggy_code.py",
        "language":"","line_count":0,"functions":[],
        "security_issues":[],"bug_issues":[],"quality_issues":[],
        "all_issues":[],"risk_score":0,"risk_label":"","final_report":"",
        "needs_human_review":False,"human_approved":None,
        "token_usage":{},"processing_steps":[],"elapsed_ms":{},
        "error":None,"retry_count":0
    }
    print("── Structured logs appear as each node runs ──\n")
    result = app.invoke(state, config={"configurable":{"thread_id":"demo"}})
    print("\n── Final Report ──\n")
    print(result["final_report"])
