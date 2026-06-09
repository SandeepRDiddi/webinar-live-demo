"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 3 — FIRST LANGGRAPH NODE: parse_code
 "One node. One job. Testable alone."
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 What's new:
   ✦ First LangGraph node — a pure function (state in, state out)
   ✦ Claude returns structured JSON (not free text)
   ✦ Graph with ONE node — already resumable + checkpointed

 Run: python step3_first_node.py

 KEY INSIGHT: A node is just a function.
   def parse_code(state: ReviewState) -> dict:
       ...
       return {"language": "Python", "line_count": 80, ...}
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


# ── State (same as step 2) ──────────────────────────────────────────
class ReviewState(TypedDict):
    code: str;         filename: str
    language: str;     line_count: int;    functions: List[str]
    security_issues: List[dict];           bug_issues: List[dict]
    quality_issues: List[dict];            all_issues: List[dict]
    risk_score: int;   risk_label: str;    final_report: str
    needs_human_review: bool;              human_approved: Optional[bool]
    token_usage: dict; processing_steps: List[str]; elapsed_ms: dict
    error: Optional[str];                  retry_count: int


# ── NODE 1: parse_code ──────────────────────────────────────────────
# A node is just a function: takes state, returns a dict of updates.
# LangGraph merges the returned dict into the state automatically.

def parse_code(state: ReviewState) -> dict:
    print(f"  [parse_code] Analysing {state['filename']}...")
    t0 = time.time()

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=512,
        system="You are a code analyser. Return ONLY valid JSON, no markdown.",
        messages=[{
            "role": "user",
            "content": f"""Analyse this code and return JSON:
{{
  "language": "<detected language>",
  "line_count": <number>,
  "functions": ["list", "of", "function", "names"]
}}

CODE:
{state['code']}"""
        }]
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    data = json.loads(raw)
    elapsed = (time.time() - t0) * 1000
    tokens = response.usage.input_tokens + response.usage.output_tokens

    return {
        "language":          data.get("language", "unknown"),
        "line_count":        data.get("line_count", 0),
        "functions":         data.get("functions", []),
        "processing_steps":  state.get("processing_steps", []) + ["parse_code"],
        "token_usage":       {**state.get("token_usage", {}), "parse_code": tokens},
        "elapsed_ms":        {**state.get("elapsed_ms", {}), "parse_code": round(elapsed, 1)},
    }


# ── Build the graph ─────────────────────────────────────────────────
def build_graph():
    workflow = StateGraph(ReviewState)
    workflow.add_node("parse_code", parse_code)   # Register node
    workflow.set_entry_point("parse_code")          # Start here
    workflow.add_edge("parse_code", END)            # Then finish
    return workflow.compile(checkpointer=MemorySaver())


# ── Run it ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = build_graph()

    initial_state: ReviewState = {
        "code": open("samples/buggy_code.py").read(),
        "filename": "buggy_code.py",
        "language": "", "line_count": 0, "functions": [],
        "security_issues": [], "bug_issues": [], "quality_issues": [],
        "all_issues": [], "risk_score": 0, "risk_label": "",
        "final_report": "", "needs_human_review": False, "human_approved": None,
        "token_usage": {}, "processing_steps": [], "elapsed_ms": {},
        "error": None, "retry_count": 0
    }

    print("Running 1-node LangGraph pipeline...\n")
    result = app.invoke(initial_state, config={"configurable": {"thread_id": "demo"}})

    print(f"\n✅ parse_code complete:")
    print(f"   Language   : {result['language']}")
    print(f"   Lines      : {result['line_count']}")
    print(f"   Functions  : {result['functions']}")
    print(f"   Tokens used: {result['token_usage'].get('parse_code', 0)}")
    print(f"   Time       : {result['elapsed_ms'].get('parse_code', 0)}ms")
    print(f"\n👆 One node. Structured output. Independently testable.")
    print(f"   Next: add a security scanner node.")
