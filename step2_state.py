"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 2 — TYPED STATE: THE FOUNDATION
 "Give the agent memory."
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 What's new:
   ✦ TypedDict state — every node reads/writes the same object
   ✦ Pydantic CodeIssue — every finding has severity, line, fix
   ✦ No more string blobs — data we can filter, sort, alert on

 Run: python step2_state.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from typing import TypedDict, List, Optional
from pydantic import BaseModel
from enum import Enum
import json


# ── The shape of every finding ─────────────────────────────────────
class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"

class CodeIssue(BaseModel):
    severity:    Severity
    category:    str           # "security" | "bug" | "quality"
    line_number: Optional[int]
    title:       str
    description: str
    suggestion:  str
    code_fix:    Optional[str] = None


# ── The single state object that flows through every node ───────────
class ReviewState(TypedDict):
    # INPUT
    code:     str
    filename: str

    # Set by parse_code node
    language:   str
    line_count: int
    functions:  List[str]

    # Set by scanner nodes
    security_issues: List[dict]
    bug_issues:      List[dict]
    quality_issues:  List[dict]

    # Set by aggregate node
    all_issues:  List[dict]
    risk_score:  int           # 0–100
    risk_label:  str           # CRITICAL / HIGH / MEDIUM / LOW

    # Set by report node
    final_report: str

    # Production controls
    needs_human_review: bool
    human_approved:     Optional[bool]

    # Observability
    token_usage:       dict   # {"node_name": tokens_used}
    processing_steps:  List[str]
    elapsed_ms:        dict   # {"node_name": ms}

    # Error handling
    error:       Optional[str]
    retry_count: int


# ── Demo: build a state and show it ────────────────────────────────
if __name__ == "__main__":
    # What the initial state looks like before any node runs
    initial_state: ReviewState = {
        "code":     open("samples/buggy_code.py").read(),
        "filename": "buggy_code.py",
        "language": "",
        "line_count": 0,
        "functions": [],
        "security_issues": [],
        "bug_issues": [],
        "quality_issues": [],
        "all_issues": [],
        "risk_score": 0,
        "risk_label": "",
        "final_report": "",
        "needs_human_review": False,
        "human_approved": None,
        "token_usage": {},
        "processing_steps": [],
        "elapsed_ms": {},
        "error": None,
        "retry_count": 0
    }

    print("✅ State initialised. Shape:")
    for key, val in initial_state.items():
        print(f"   {key:25} → {type(val).__name__}")

    print("\n✅ What a finding looks like (Pydantic):")
    sample_issue = CodeIssue(
        severity=Severity.CRITICAL,
        category="security",
        line_number=10,
        title="SQL Injection",
        description="f-string directly in SQL query allows attackers to inject arbitrary SQL",
        suggestion="Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
        code_fix="cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))"
    )
    print(json.dumps(sample_issue.model_dump(), indent=2))
    print("\n👆 Severity is typed. Line number is exact. Code fix is ready to copy.")
    print("   THIS is the difference between a vibe output and a production output.")
