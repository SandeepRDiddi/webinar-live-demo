"""
LangGraph Code Review Agent — Streamlit Webinar Demo
Step-by-step visual journey: vibe-coding → production agent.
"""

import streamlit as st
import anthropic
import json
import time
import os
from typing import TypedDict, List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="LangGraph Agent: Webinar Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────
st.markdown("""<style>
.analogy-box {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d1b69 100%);
    color: #e2e8f0;
    padding: 20px 24px;
    border-radius: 14px;
    margin: 14px 0 20px 0;
    border-left: 5px solid #60a5fa;
}
.analogy-box h3 { color: #93c5fd; margin-top: 0; }
.analogy-box strong { color: #fbbf24; }
.node-pill {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 0.82em;
    margin: 2px;
}
.pipeline-wrap {
    background: #0f172a;
    border-radius: 12px;
    padding: 18px 20px;
    overflow-x: auto;
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
}
.pnode {
    padding: 10px 14px;
    border-radius: 10px;
    text-align: center;
    min-width: 110px;
    font-family: monospace;
    font-size: 12px;
    line-height: 1.5;
    font-weight: 600;
}
.pnode-pending  { background: #1e293b; color: #64748b; border: 1px solid #334155; }
.pnode-active   { background: #1d4ed8; color: white; border: 2px solid #60a5fa; }
.pnode-done     { background: #166534; color: #bbf7d0; border: 1px solid #22c55e; }
.pnode-hitl     { background: #92400e; color: #fde68a; border: 2px solid #f59e0b; }
.arrow { color: #475569; font-size: 20px; }
.log-entry-INFO  { border-left: 3px solid #3b82f6; }
.log-entry-WARN  { border-left: 3px solid #f59e0b; }
.log-entry-ERROR { border-left: 3px solid #ef4444; }
.log-entry {
    background: #0f172a;
    color: #94a3b8;
    font-family: monospace;
    font-size: 12px;
    padding: 6px 10px;
    margin: 3px 0;
    border-radius: 4px;
}
</style>""", unsafe_allow_html=True)

# ── Buggy code sample ─────────────────────────────────────────────────
_sample_path = os.path.join(os.path.dirname(__file__), "samples", "buggy_code.py")
DEFAULT_CODE = open(_sample_path).read() if os.path.exists(_sample_path) else "# paste code here"

# ── Anthropic client ──────────────────────────────────────────────────
from agent_prism import make_client, report_review, MODEL

@st.cache_resource
def get_client():
    return make_client()

client = get_client()

# ── State definition ──────────────────────────────────────────────────
class ReviewState(TypedDict):
    code: str;              filename: str
    language: str;          line_count: int;    functions: List[str]
    security_issues: List[dict];               bug_issues: List[dict]
    quality_issues: List[dict];                all_issues: List[dict]
    risk_score: int;        risk_label: str;    final_report: str
    needs_human_review: bool;                  human_approved: Optional[bool]
    token_usage: dict;      processing_steps: List[str];  elapsed_ms: dict
    error: Optional[str];   retry_count: int

def empty_state(code: str, filename: str = "buggy_code.py") -> ReviewState:
    return {
        "code": code, "filename": filename, "language": "", "line_count": 0,
        "functions": [], "security_issues": [], "bug_issues": [], "quality_issues": [],
        "all_issues": [], "risk_score": 0, "risk_label": "", "final_report": "",
        "needs_human_review": False, "human_approved": None,
        "token_usage": {}, "processing_steps": [], "elapsed_ms": {},
        "error": None, "retry_count": 0,
    }

# ── Claude helper ─────────────────────────────────────────────────────
def call_claude(system: str, user: str, log_sink=None, max_tokens: int = 4096) -> Tuple[dict, int, float]:
    for attempt in range(1, 4):
        try:
            t0 = time.time()
            r = client.messages.create(
                model=MODEL, max_tokens=max_tokens,
                system=system, messages=[{"role": "user", "content": user}]
            )
            text = r.content[0].text.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:-1])
            data = json.loads(text)
            elapsed = round((time.time() - t0) * 1000, 1)
            tokens = r.usage.input_tokens + r.usage.output_tokens
            return data, tokens, elapsed
        except json.JSONDecodeError:
            if log_sink is not None:
                log_sink.append({"ts": _ts(), "level": "WARN", "event": "json_retry",
                                 "attempt": attempt})
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)

def _ts():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%H:%M:%S")

# ── Pipeline nodes ────────────────────────────────────────────────────
def node_parse_code(state, logs=None):
    d, tok, ms = call_claude(
        "Return ONLY valid JSON, no markdown.",
        f'Analyse this code. Return exactly: {{"language":"<lang>","line_count":<n>,"functions":["fn1","fn2"]}}\n\nCODE:\n{state["code"]}',
        logs
    )
    if logs is not None:
        logs.append({"ts": _ts(), "level": "INFO", "event": "node_completed",
                     "node": "parse_code", "ms": ms, "tokens": tok,
                     "language": d.get("language"), "line_count": d.get("line_count")})
    return {
        "language": d.get("language", "unknown"), "line_count": d.get("line_count", 0),
        "functions": d.get("functions", []),
        "processing_steps": state.get("processing_steps", []) + ["parse_code"],
        "token_usage": {**state.get("token_usage", {}), "parse_code": tok},
        "elapsed_ms": {**state.get("elapsed_ms", {}), "parse_code": ms},
    }

def node_security_scan(state, logs=None):
    d, tok, ms = call_claude(
        "Senior security engineer. Return ONLY valid JSON, no markdown.",
        f"""Find ALL security vulnerabilities in this {state['language']} code.
Return JSON: {{"issues":[{{"severity":"critical|high|medium|low","line_number":<n_or_null>,"title":"","description":"","suggestion":"","code_fix":null,"owasp":""}}],"has_critical":false}}
CODE:\n{state['code']}""",
        logs
    )
    issues = d.get("issues", [])
    if logs is not None:
        logs.append({"ts": _ts(), "level": "INFO", "event": "node_completed",
                     "node": "security_scan", "ms": ms, "tokens": tok,
                     "issues_found": len(issues),
                     "has_critical": d.get("has_critical", False)})
    return {
        "security_issues": issues,
        "processing_steps": state.get("processing_steps", []) + ["security_scan"],
        "token_usage": {**state.get("token_usage", {}), "security_scan": tok},
        "elapsed_ms": {**state.get("elapsed_ms", {}), "security_scan": ms},
    }

def node_bug_detection(state, logs=None):
    d, tok, ms = call_claude(
        "Senior software engineer. Return ONLY valid JSON, no markdown.",
        f"""Find runtime bugs in this {state['language']} code.
Return JSON: {{"issues":[{{"severity":"critical|high|medium|low","line_number":<n_or_null>,"title":"","description":"","suggestion":"","code_fix":null}}]}}
CODE:\n{state['code']}""",
        logs
    )
    issues = d.get("issues", [])
    if logs is not None:
        logs.append({"ts": _ts(), "level": "INFO", "event": "node_completed",
                     "node": "bug_detection", "ms": ms, "tokens": tok,
                     "issues_found": len(issues)})
    return {
        "bug_issues": issues,
        "processing_steps": state.get("processing_steps", []) + ["bug_detection"],
        "token_usage": {**state.get("token_usage", {}), "bug_detection": tok},
        "elapsed_ms": {**state.get("elapsed_ms", {}), "bug_detection": ms},
    }

def node_aggregate(state, logs=None):
    all_i = (state.get("security_issues", []) + state.get("bug_issues", [])
             + state.get("quality_issues", []))
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for i in all_i:
        s = i.get("severity", "low").lower()
        counts[s] = counts.get(s, 0) + 1
    score = min(100, counts["critical"]*25 + counts["high"]*10
                + counts["medium"]*5 + counts["low"])
    label = ("CRITICAL" if score >= 75 else "HIGH" if score >= 50
             else "MEDIUM" if score >= 25 else "LOW")
    needs = counts["critical"] > 0
    if logs is not None:
        lvl = "WARN" if needs else "INFO"
        ev  = "human_review_required" if needs else "node_completed"
        logs.append({"ts": _ts(), "level": lvl, "event": ev,
                     "node": "aggregate_issues", "risk_score": score,
                     "label": label, "total_issues": len(all_i),
                     "critical": counts["critical"]})
    return {
        "all_issues": all_i, "risk_score": score, "risk_label": label,
        "needs_human_review": needs,
        "processing_steps": state.get("processing_steps", []) + ["aggregate_issues"],
    }

def node_generate_report(state, human_approved=None, logs=None):
    all_i = state.get("all_issues", [])
    counts = {}
    for i in all_i:
        s = i.get("severity", "low"); counts[s] = counts.get(s, 0) + 1
    so = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    sorted_i = sorted(all_i, key=lambda x: so.get(x.get("severity", "low"), 4))
    report = (f"# Code Review: {state['filename']}\n"
              f"Risk: {state['risk_score']}/100 — {state['risk_label']}\n"
              f"Issues: {len(all_i)} total  "
              f"({counts.get('critical',0)} critical · {counts.get('high',0)} high · "
              f"{counts.get('medium',0)} medium · {counts.get('low',0)} low)\n\n")
    for idx, issue in enumerate(sorted_i, 1):
        ln = f" (line {issue['line_number']})" if issue.get("line_number") else ""
        report += f"[{idx}] [{issue.get('severity','').upper()}] {issue.get('title','')}{ln}\n"
        report += f"     {issue.get('description', '')}\n"
        report += f"     FIX: {issue.get('suggestion','')}\n\n"
    total_tok = sum(state.get("token_usage", {}).values())
    total_ms  = sum(state.get("elapsed_ms", {}).values())
    if logs is not None:
        logs.append({"ts": _ts(), "level": "INFO", "event": "review_complete",
                     "file": state["filename"], "risk_score": state["risk_score"],
                     "total_issues": len(all_i), "total_ms": round(total_ms, 1),
                     "total_tokens": total_tok})
    result = {
        "final_report": report,
        "processing_steps": state.get("processing_steps", []) + ["generate_report"],
    }
    if human_approved is not None:
        result["human_approved"] = human_approved
    return result

# ── Pipeline graph HTML ───────────────────────────────────────────────
def pipeline_html(completed=None, active=None, hitl_active=False, show_hitl=True):
    completed = completed or set()
    active    = active    or set()

    def cls(name):
        if name in active:   return "pnode pnode-active"
        if hitl_active and name == "human_review_gate": return "pnode pnode-hitl"
        if name in completed: return "pnode pnode-done"
        return "pnode pnode-pending"

    nodes = [
        ("parse_code",         "📋 parse_code<br><small>detect language</small>"),
        ("security_scan",      "🔒 security_scan<br><small>OWASP Top 10</small>"),
        ("bug_detection",      "🐛 bug_detection<br><small>runtime bugs</small>"),
        ("aggregate_issues",   "📊 aggregate<br><small>risk score</small>"),
    ]
    html = '<div class="pipeline-wrap">'
    for i, (name, label) in enumerate(nodes):
        if i > 0:
            html += '<span class="arrow">→</span>'
        html += f'<div class="{cls(name)}">{label}</div>'

    # Fork after aggregate
    if show_hitl:
        html += '<span class="arrow">↗</span>'
        html += f'<div class="{cls("human_review_gate")}" style="background:#7c2d12;color:#fed7aa;border:1px solid #ea580c;">🧑‍💼 human_gate<br><small>if critical</small></div>'
        html += '<span class="arrow">↘</span>'
    else:
        html += '<span class="arrow">→</span>'

    html += f'<div class="{cls("generate_report")}">📄 generate_report<br><small>final output</small></div>'
    html += '</div>'
    return html

# ── Severity helpers ──────────────────────────────────────────────────
SEV_COLOR = {
    "critical": ("#FF4444", "#FFE5E5", "🔴"),
    "high":     ("#EF6C00", "#FFF3E5", "🟠"),
    "medium":   ("#F9A825", "#FFFDE5", "🟡"),
    "low":      ("#2E7D32", "#E8F5E9", "🟢"),
}

def issue_card(issue: dict, idx: int):
    sev  = issue.get("severity", "low").lower()
    col, bg, emoji = SEV_COLOR.get(sev, ("#888", "#F5F5F5", "⚪"))
    line = f" · line {issue['line_number']}" if issue.get("line_number") else ""
    expanded = sev == "critical"
    with st.expander(f"{emoji} [{sev.upper()}] {issue.get('title','Untitled')}{line}",
                     expanded=expanded):
        st.markdown(f"**What:** {issue.get('description', '')}")
        st.markdown(f"**Fix:** {issue.get('suggestion', '')}")
        if issue.get("owasp"):
            st.caption(f"OWASP ref: {issue['owasp']}")
        if issue.get("code_fix"):
            st.code(issue["code_fix"], language="python")

def risk_bar(score: int, label: str):
    color = "#FF4444" if score >= 75 else "#EF6C00" if score >= 50 else "#F9A825" if score >= 25 else "#2E7D32"
    st.markdown(f"""
<div style="background:#1e293b;border-radius:10px;padding:16px 20px;margin:8px 0;">
  <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
    <span style="color:#94a3b8;font-size:14px;">Risk Score</span>
    <span style="color:{color};font-weight:bold;font-size:18px;">{score}/100 — {label}</span>
  </div>
  <div style="background:#334155;border-radius:6px;height:16px;overflow:hidden;">
    <div style="background:{color};width:{score}%;height:100%;border-radius:6px;
                transition:width 0.5s ease;"></div>
  </div>
</div>""", unsafe_allow_html=True)

def issue_breakdown_bar(all_issues):
    if not all_issues:
        return
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for i in all_issues:
        s = i.get("severity", "low").lower()
        counts[s] = counts.get(s, 0) + 1
    total = sum(counts.values()) or 1
    st.markdown("**Issues by severity**")
    for sev, cnt in counts.items():
        if cnt == 0:
            continue
        col, _, emoji = SEV_COLOR.get(sev, ("#888", "#F5F5F5", "⚪"))
        pct = int(cnt / total * 100)
        st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;margin:4px 0;">
  <span style="width:60px;color:{col};font-weight:bold;font-size:13px;">{emoji} {sev}</span>
  <div style="flex:1;background:#1e293b;border-radius:4px;height:20px;">
    <div style="background:{col};width:{pct}%;height:100%;border-radius:4px;"></div>
  </div>
  <span style="width:24px;text-align:right;color:#94a3b8;font-size:13px;">{cnt}</span>
</div>""", unsafe_allow_html=True)

def perf_metrics(token_usage, elapsed_ms):
    if not token_usage:
        return
    total_tok = sum(token_usage.values())
    total_ms  = sum(elapsed_ms.values())
    cols = st.columns(len(token_usage) + 2)
    for i, (node, tok) in enumerate(token_usage.items()):
        ms = elapsed_ms.get(node, 0)
        cols[i].metric(node.replace("_", " "), f"{tok:,} tok", f"{ms:.0f}ms")
    cols[-2].metric("Total tokens", f"{total_tok:,}")
    cols[-1].metric("Total time",   f"{total_ms/1000:.1f}s")

def log_display(logs):
    for entry in logs:
        lvl = entry.get("level", "INFO")
        css = f"log-entry log-entry-{lvl}"
        text = json.dumps(entry, indent=None)
        st.markdown(f'<div class="{css}">{text}</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════
#  STEP RENDERERS
# ═════════════════════════════════════════════════════════════════════

def show_step1():
    st.header("Step 1 — The Vibe Coded Version 🤷")
    st.caption("*'It works. Ship it.'*  |  The starting point for every developer")

    st.markdown("""<div class="analogy-box">
<h3>🏥 Real World: The Chatty Doctor</h3>
You ask your doctor <em>"Am I healthy?"</em> — and they talk for 3 minutes straight.
No diagnosis form. No severity rating. No prescription. Just vibes.<br><br>
You can't search it. Can't sort it. Can't build an alert on it.
Can't tell your nurse what to do next. <strong>That's exactly what this code does.</strong>
</div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### The code — 8 lines, ships Friday")
        st.code("""def review_code(code: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"Review this code:\\n{code}"
        }]
    )
    return response.content[0].text   # raw blob""", language="python")

        st.warning("Returns unstructured text. You cannot filter, sort, alert, or build on it.")

    with col2:
        st.markdown("#### Questions you CANNOT answer:")
        for q in [
            "How many CRITICAL issues exist?",
            "Which exact line has the SQL injection?",
            "What's the overall risk score (0–100)?",
            "Can I sort issues by severity?",
            "Can I trigger a Slack alert on critical findings?",
        ]:
            st.error(f"❌ {q}")

    st.divider()
    code_input = st.text_area("Code to review:", value=DEFAULT_CODE, height=180, key="s1_code")

    if st.button("▶ Run Vibe Review", type="primary", key="s1_run"):
        with st.spinner("Calling Claude (raw, unstructured)..."):
            try:
                t0 = time.time()
                r = client.messages.create(
                    model=MODEL, max_tokens=1024,
                    messages=[{"role": "user",
                               "content": f"Review this code and find issues:\n\n{code_input}"}]
                )
                blob = r.content[0].text
                elapsed = (time.time() - t0) * 1000
                tokens = r.usage.input_tokens + r.usage.output_tokens
            except Exception as e:
                st.error(f"API error: {e}")
                return

        st.markdown("#### 📤 Output — a raw blob of text")
        st.text_area("", value=blob, height=320, disabled=True, key="s1_out")

        c1, c2, c3 = st.columns(3)
        c1.metric("Tokens used", f"{tokens:,}")
        c2.metric("Time", f"{elapsed:.0f} ms")
        c3.metric("Structured?", "❌ No")

        st.error("""**Now try to answer the questions above.**
You can't. This is the problem Steps 2–7 solve.""")

        st.info("""**The gap:** Not whether Claude finds issues — it does.
The gap is whether you can **do anything with the output at scale**.
Filter it. Sort it. Route it. Alert on it. Feed it to the next node. You can't. Yet.""")


def show_step2():
    st.header("Step 2 — Typed State: The Foundation 📋")
    st.caption("*'Give the agent memory.'*  |  Every node reads and writes the same typed object")

    st.markdown("""<div class="analogy-box">
<h3>🏥 Real World: The Patient Medical Record</h3>
Instead of a doctor's scribbled notes, every piece of information has a <em>typed field</em>.<br>
Blood pressure: <strong>120</strong> (not "kinda high"). Temperature: <strong>98.6</strong>. Allergies: <strong>["penicillin"]</strong>.<br><br>
The nurse, the lab, the pharmacist, and the specialist all read the <strong>same structured record</strong>.
Every LangGraph node does exactly this — reads from one shared state, writes back typed updates.
</div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### `ReviewState` — the shared form")
        rows = [
            ("code",               "str",          "INPUT — the file"),
            ("filename",           "str",          "INPUT — name"),
            ("language",           "str",          "← parse_code"),
            ("line_count",         "int",          "← parse_code"),
            ("functions",          "List[str]",    "← parse_code"),
            ("security_issues",    "List[dict]",   "← security_scan"),
            ("bug_issues",         "List[dict]",   "← bug_detection"),
            ("all_issues",         "List[dict]",   "← aggregate"),
            ("risk_score",         "int 0-100",    "← aggregate"),
            ("risk_label",         "str",          "← aggregate"),
            ("needs_human_review", "bool",         "← aggregate"),
            ("human_approved",     "Optional[bool]","← human_gate"),
            ("token_usage",        "dict",         "observability"),
            ("elapsed_ms",         "dict",         "observability"),
        ]
        st.markdown("| Field | Type | Set by |")
        st.markdown("|---|---|---|")
        for field, typ, who in rows:
            st.markdown(f"| `{field}` | `{typ}` | *{who}* |")

    with col2:
        st.markdown("#### `CodeIssue` — one finding, fully typed")
        sample = {
            "severity": "critical",
            "category": "security",
            "line_number": 37,
            "title": "SQL Injection",
            "description": "f-string in SQL allows arbitrary query injection",
            "suggestion": "Use parameterized queries: cursor.execute(..., (email,))",
            "code_fix": "cursor.execute('SELECT * FROM users WHERE email = ?', (email,))"
        }
        st.json(sample)
        st.success("✅ Severity is an enum. Line number is exact. Code fix is ready to copy-paste.")

        st.divider()
        st.markdown("#### The transformation")
        c1, c2 = st.columns(2)
        c1.error("**Step 1 output**\n`str` → free text blob")
        c2.success("**Step 2 output**\n`ReviewState` → typed, sortable, filterable, alertable")

        st.markdown("""#### Why this matters
- Filter by `severity == 'critical'`
- Sort issues by `line_number`
- Count issues in a category
- Store in Postgres, send to Slack
- Feed to the next node — typed

**You can't do ANY of this with a text blob.**""")

    with st.expander("▶ Run: Show state structure"):
        state = empty_state(DEFAULT_CODE)
        st.markdown("**Initial state shape (before any node runs):**")
        shape = {k: type(v).__name__ for k, v in state.items()}
        st.json(shape)
        st.info("Every node reads from this. Every node writes typed updates back to it.")


def show_step3():
    st.header("Step 3 — First LangGraph Node: parse_code 🔧")
    st.caption("*'One node. One job. Independently testable.'*")

    st.markdown("""<div class="analogy-box">
<h3>🏥 Real World: The Hospital Receptionist</h3>
The receptionist does <strong>ONE job</strong>: check you in, identify what kind of patient you are,
and fill out the intake form. They don't diagnose. They don't prescribe.<br><br>
If the receptionist makes a mistake, you know <strong>exactly where to look</strong>.
Independently testable. Independently replaceable.
That's what a LangGraph node is.
</div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### A node is just a function")
        st.code("""def parse_code(state: ReviewState) -> dict:
    # Read from state
    code = state["code"]

    # Call Claude with structured prompt
    response = call_claude(
        system="Return ONLY valid JSON.",
        user=f'Return {{"language":"...","line_count":n,"functions":["..."]}}'
             f'\\nCODE:\\n{code}'
    )

    # Return ONLY the fields this node owns
    return {
        "language":   "Python",
        "line_count": 80,
        "functions":  ["login", "get_user"]
    }
# LangGraph MERGES this dict into state automatically""", language="python")

        st.markdown("#### Graph: 1 node")
        st.markdown(pipeline_html(show_hitl=False), unsafe_allow_html=True)
        st.caption("START → parse_code → END   |  Already resumable. Already checkpointed.")

    with col2:
        st.markdown("#### Run parse_code")
        code_input = st.text_area("Code:", value=DEFAULT_CODE, height=140, key="s3_code")

        if st.button("▶ Run parse_code node", type="primary", key="s3_run"):
            graph_ph = st.empty()
            graph_ph.markdown(pipeline_html(active={"parse_code"}, show_hitl=False),
                              unsafe_allow_html=True)
            with st.status("Running parse_code...", expanded=True) as status:
                st.write("📋 Asking Claude to detect language, count lines, list functions...")
                try:
                    state = empty_state(code_input)
                    result = node_parse_code(state)
                    state.update(result)
                    status.update(label="✅ parse_code complete!", state="complete")
                except Exception as e:
                    status.update(label=f"❌ {e}", state="error")
                    st.error(str(e))
                    return

            graph_ph.markdown(pipeline_html(completed={"parse_code"}, show_hitl=False),
                              unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Language",  state["language"])
            c2.metric("Lines",     state["line_count"])
            c3.metric("Functions", len(state["functions"]))

            if state["functions"]:
                st.markdown("**Detected functions:**")
                st.code(", ".join(state["functions"]))

            tok = result["token_usage"].get("parse_code", 0)
            ms  = result["elapsed_ms"].get("parse_code", 0)
            st.info(f"⚡ {tok:,} tokens · {ms:.0f}ms")
            st.success("One node. Structured output. Independently testable. Next: add a security specialist.")


def show_step4():
    st.header("Step 4 — Security Scanner Node 🔒")
    st.caption("*'A specialist joins the team.'*  |  OWASP Top 10 check")

    st.markdown("""<div class="analogy-box">
<h3>🏥 Real World: The Infectious Disease Specialist</h3>
After the receptionist fills out your intake form,
an <em>infectious disease specialist</em> reviews your chart for dangerous conditions.<br><br>
This specialist knows exactly what to look for: <strong>OWASP Top 10</strong> —
the 10 most common ways software gets hacked.
SQL Injection. Broken Authentication. Hardcoded Secrets. Logged Passwords.
They check all of them, and return <strong>structured findings with exact line numbers</strong>.
</div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("#### Graph: parse → security → END")
        st.markdown(pipeline_html(show_hitl=False), unsafe_allow_html=True)

    with col2:
        st.markdown("#### What changed")
        st.code("""# Two new lines in build_graph():
workflow.add_node(
    "security_scan",
    security_scan      # ← specialist fn
)
workflow.add_edge(
    "parse_code",
    "security_scan"    # ← wire it in
)""", language="python")
        st.success("**Two lines to add a specialist.** That's the LangGraph pattern.")

    st.markdown("#### What the security scanner returns")
    sample_finding = {
        "severity": "critical",
        "line_number": 37,
        "title": "SQL Injection via f-string",
        "description": "Direct string interpolation in SQL allows attackers to inject arbitrary queries",
        "suggestion": "Use parameterized queries: cursor.execute('SELECT * FROM users WHERE email = ?', (email,))",
        "code_fix": "cursor.execute('SELECT * FROM users WHERE email = ?', (email,))",
        "owasp": "A03:2021 Injection"
    }

    col1, col2 = st.columns([1, 1])
    with col1:
        st.json(sample_finding)
    with col2:
        st.markdown("Compare to Step 1 (vibe):")
        st.error("**Step 1:** 'you might have an injection issue somewhere...'")
        st.success("**Step 4:** severity=CRITICAL · line=37 · OWASP ref · code fix included")

    st.divider()
    code_input = st.text_area("Code:", value=DEFAULT_CODE, height=120, key="s4_code")

    if st.button("▶ Run parse → security pipeline", type="primary", key="s4_run"):
        graph_ph = st.empty()
        result_ph = st.container()

        graph_ph.markdown(pipeline_html(active={"parse_code"}, show_hitl=False), unsafe_allow_html=True)

        with st.status("Running 2-node pipeline...", expanded=True) as status:
            try:
                state = empty_state(code_input)
                completed = set()

                st.write("📋 Node 1/2: parse_code...")
                result = node_parse_code(state)
                state.update(result)
                completed.add("parse_code")
                st.write(f"   → {state['language']}, {state['line_count']} lines, {len(state['functions'])} functions")
                graph_ph.markdown(pipeline_html(completed=completed, active={"security_scan"},
                                                show_hitl=False), unsafe_allow_html=True)

                st.write("🔒 Node 2/2: security_scan...")
                result = node_security_scan(state)
                state.update(result)
                completed.add("security_scan")
                st.write(f"   → {len(state['security_issues'])} security issues found")
                graph_ph.markdown(pipeline_html(completed=completed, show_hitl=False),
                                  unsafe_allow_html=True)

                status.update(label=f"✅ Done — {len(state['security_issues'])} security issues found",
                              state="complete")
            except Exception as e:
                status.update(label=f"❌ {e}", state="error")
                st.error(str(e))
                return

        with result_ph:
            st.divider()
            st.markdown(f"### 🔒 Security Findings ({len(state['security_issues'])} total)")
            if state["security_issues"]:
                for i, issue in enumerate(state["security_issues"]):
                    issue_card(issue, i)
            else:
                st.success("No security vulnerabilities found.")

            perf_metrics(state["token_usage"], state["elapsed_ms"])


def show_step5():
    st.header("Step 5 — Full Pipeline + Human-in-the-Loop 🏭")
    st.caption("*'5 nodes. Risk score. Nuclear launch protocol.'*")

    st.markdown("""<div class="analogy-box">
<h3>🏥 Real World: The Nuclear Launch Protocol</h3>
Before launching, <strong>multiple specialists review</strong>, AND a human commander
must physically turn a key to approve it. No single person — and no AI — can do it alone.<br><br>
Our agent does exactly this: when it finds <strong>CRITICAL vulnerabilities</strong>,
it <em>PAUSES and WAITS</em> for a human to approve before generating the final report.
No critical code ships without human eyes. That's <strong>Human-in-the-Loop (HITL)</strong>.
</div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("#### Full graph with conditional routing")
        st.markdown(pipeline_html(), unsafe_allow_html=True)
    with col2:
        st.markdown("#### The routing function")
        st.code("""def should_escalate(state):
    if state["needs_human_review"]:
        return "escalate"    # → HITL
    return "report"          # → direct""", language="python")
        st.info("One function. Returns a string. LangGraph routes accordingly.")

    # ── Session state init ────────────────────────────────────────────
    for k, v in [("s5_state", None), ("s5_hitl_needed", False), ("s5_done", False)]:
        if k not in st.session_state:
            st.session_state[k] = v

    code_input = st.text_area("Code:", value=DEFAULT_CODE, height=100, key="s5_code")
    graph_ph   = st.empty()
    graph_ph.markdown(pipeline_html(), unsafe_allow_html=True)

    if st.button("▶ Run Full Pipeline", type="primary", key="s5_run",
                 disabled=st.session_state.s5_hitl_needed):

        # Reset
        st.session_state.s5_state = None
        st.session_state.s5_hitl_needed = False
        st.session_state.s5_done = False

        NODES = [
            ("parse_code",       node_parse_code,   "📋"),
            ("security_scan",    node_security_scan, "🔒"),
            ("bug_detection",    node_bug_detection, "🐛"),
            ("aggregate_issues", node_aggregate,     "📊"),
        ]
        with st.status("Running full pipeline...", expanded=True) as status:
            try:
                state    = empty_state(code_input)
                completed = set()

                for node_name, node_fn, emoji in NODES:
                    graph_ph.markdown(pipeline_html(completed=completed, active={node_name}),
                                      unsafe_allow_html=True)
                    st.write(f"{emoji} Running {node_name}...")
                    result = node_fn(state)
                    state.update(result)
                    completed.add(node_name)
                    if node_name == "security_scan":
                        st.write(f"   → {len(state['security_issues'])} security issues")
                    elif node_name == "bug_detection":
                        st.write(f"   → {len(state['bug_issues'])} bugs")
                    elif node_name == "aggregate_issues":
                        st.write(f"   → Risk: {state['risk_score']}/100 ({state['risk_label']})")

                st.session_state.s5_state = state

                if state["needs_human_review"]:
                    st.session_state.s5_hitl_needed = True
                    graph_ph.markdown(pipeline_html(completed=completed, hitl_active=True),
                                      unsafe_allow_html=True)
                    status.update(label="⏸ PAUSED — Human Review Required!", state="running")
                else:
                    result = node_generate_report(state)
                    state.update(result)
                    completed.add("generate_report")
                    graph_ph.markdown(pipeline_html(completed=completed), unsafe_allow_html=True)
                    st.session_state.s5_done = True
                    st.session_state.s5_state = state
                    status.update(label="✅ Pipeline complete!", state="complete")
                    report_review(state, state.get("filename", "code.py"))
            except Exception as e:
                status.update(label=f"❌ {e}", state="error")
                st.error(str(e))

    # ── HITL gate ─────────────────────────────────────────────────────
    if st.session_state.s5_hitl_needed and not st.session_state.s5_done:
        state = st.session_state.s5_state
        st.divider()
        st.error("## 🚨 AGENT PAUSED — HUMAN REVIEW REQUIRED")
        st.markdown("> *The agent found CRITICAL vulnerabilities and cannot proceed without your approval.*")

        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Score",      f"{state['risk_score']}/100")
        c2.metric("Risk Label",      state['risk_label'])
        critical_n = sum(1 for i in state['all_issues'] if i.get('severity') == 'critical')
        c3.metric("Critical Issues", critical_n)

        st.markdown("**Critical findings requiring your decision:**")
        for issue in state.get("all_issues", []):
            if issue.get("severity") == "critical":
                issue_card(issue, 0)

        st.divider()
        c1, c2 = st.columns(2)
        approve = c1.button("✅  APPROVE — Generate Report", type="primary", key="s5_approve")
        reject  = c2.button("❌  REJECT — Block This Code",               key="s5_reject")

        if approve or reject:
            approved = approve
            state["human_approved"] = approved
            state["processing_steps"].append("human_review_gate")
            r = node_generate_report(state, human_approved=approved)
            state.update(r)
            if not approved:
                state["final_report"] = "REJECTED by human reviewer. Code blocked from proceeding."
            st.session_state.s5_state = state
            st.session_state.s5_hitl_needed = False
            st.session_state.s5_done = True
            report_review(state, state.get("filename", "code.py"))
            st.rerun()

    # ── Final results ─────────────────────────────────────────────────
    if st.session_state.s5_done and st.session_state.s5_state:
        state = st.session_state.s5_state
        st.divider()
        st.markdown("### 📊 Results Dashboard")

        if state.get("human_approved") is True:
            st.success("✅ Human approved — report generated")
        elif state.get("human_approved") is False:
            st.error("❌ Human rejected — code blocked")

        col1, col2 = st.columns([1, 1])
        with col1:
            risk_bar(state["risk_score"], state["risk_label"])
        with col2:
            issue_breakdown_bar(state.get("all_issues", []))

        st.markdown(f"### All Issues ({len(state.get('all_issues',[]))} total)")
        for i, issue in enumerate(state.get("all_issues", [])):
            issue_card(issue, i)

        with st.expander("📄 Final Report (text)"):
            st.text(state["final_report"])

        st.divider()
        st.markdown("#### Performance")
        perf_metrics(state["token_usage"], state["elapsed_ms"])

        if st.button("🔄 Reset — Run Again", key="s5_reset"):
            for k in ["s5_state", "s5_hitl_needed", "s5_done"]:
                del st.session_state[k]
            st.rerun()


def show_step6():
    st.header("Step 6 — Observability 📡")
    st.caption("*'If you can't see it, you can't own it.'*  |  Structured logs on every node")

    st.markdown("""<div class="analogy-box">
<h3>🏥 Real World: Hospital CCTV + Audit Log</h3>
Every hospital room has a camera. Every doctor action is timestamped in the audit log.
When something goes wrong, you can replay <em>exactly what happened, when, and who touched the patient</em>.<br><br>
In production: <strong>Datadog, CloudWatch, Grafana</strong> all consume these structured JSON logs.
You can set an alert: <em>"page me when event='human_review_required' AND risk_score > 75"</em>.
<strong>That's a 10-minute Datadog alert rule — not a re-read of log files.</strong>
</div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### What changed from Step 5")
        st.code("""# Before (print):
print("running security scan")      # ← noise
print(f"found {n} issues")          # ← not searchable

# After (structured emit):
emit("INFO", "node_started",
     node="security_scan")

emit("INFO", "node_completed",
     node="security_scan",
     ms=1840, tokens=2103,
     issues_found=4)

emit("WARN", "human_review_required",
     file="buggy_code.py",
     risk_score=85, critical=3)""", language="python")

        st.success("Same pipeline. Same nodes. Zero logic changes. One new wrapper.")

    with col2:
        st.markdown("#### Sample log stream")
        sample_logs = [
            {"ts": "14:23:41", "level": "INFO",  "event": "node_started",            "node": "parse_code"},
            {"ts": "14:23:42", "level": "INFO",  "event": "node_completed",           "node": "parse_code",    "ms": 920,  "tokens": 410},
            {"ts": "14:23:42", "level": "INFO",  "event": "node_started",            "node": "security_scan"},
            {"ts": "14:23:44", "level": "INFO",  "event": "node_completed",           "node": "security_scan", "ms": 1840, "tokens": 2103, "issues_found": 4},
            {"ts": "14:23:44", "level": "INFO",  "event": "node_started",            "node": "bug_detection"},
            {"ts": "14:23:46", "level": "INFO",  "event": "node_completed",           "node": "bug_detection", "ms": 1620, "tokens": 1890, "issues_found": 3},
            {"ts": "14:23:46", "level": "WARN",  "event": "human_review_required",   "risk_score": 85, "critical": 3},
            {"ts": "14:23:51", "level": "INFO",  "event": "hitl_decision",           "approved": True},
            {"ts": "14:23:51", "level": "INFO",  "event": "review_complete",         "total_ms": 10200, "total_tokens": 7803},
        ]
        log_display(sample_logs)

        st.caption("🔵 INFO = normal flow  |  🟡 WARN = needs attention  |  🔴 ERROR = needs fix")

    st.markdown("#### Alert rules you can build on top of this")
    c1, c2, c3 = st.columns(3)
    c1.info("🔔 **Cost alert**\n`total_tokens > 10000` → Slack notification")
    c2.warning("⚠️ **Escalation alert**\n`event = 'human_review_required'` → PagerDuty")
    c3.error("🚨 **Failure alert**\n`event = 'node_error' AND attempt = 3` → page on-call")

    st.divider()
    code_input = st.text_area("Code:", value=DEFAULT_CODE, height=100, key="s6_code")

    # Session state
    for k, v in [("s6_logs", []), ("s6_state", None), ("s6_hitl_needed", False),
                 ("s6_done", False)]:
        if k not in st.session_state:
            st.session_state[k] = v

    graph_ph = st.empty()
    graph_ph.markdown(pipeline_html(), unsafe_allow_html=True)

    if st.button("▶ Run with Observability", type="primary", key="s6_run",
                 disabled=st.session_state.s6_hitl_needed):
        st.session_state.s6_logs = []
        st.session_state.s6_state = None
        st.session_state.s6_hitl_needed = False
        st.session_state.s6_done = False

        logs = st.session_state.s6_logs
        NODES = [
            ("parse_code",       node_parse_code,   "📋"),
            ("security_scan",    node_security_scan, "🔒"),
            ("bug_detection",    node_bug_detection, "🐛"),
            ("aggregate_issues", node_aggregate,     "📊"),
        ]
        with st.status("Running with observability...", expanded=True) as status:
            try:
                state = empty_state(code_input)
                completed = set()

                for node_name, node_fn, emoji in NODES:
                    graph_ph.markdown(pipeline_html(completed=completed, active={node_name}),
                                      unsafe_allow_html=True)
                    logs.append({"ts": _ts(), "level": "INFO", "event": "node_started",
                                 "node": node_name})
                    st.write(f"{emoji} {node_name}...")
                    result = node_fn(state, logs=logs)
                    state.update(result)
                    completed.add(node_name)

                st.session_state.s6_state = state

                if state["needs_human_review"]:
                    st.session_state.s6_hitl_needed = True
                    graph_ph.markdown(pipeline_html(completed=completed, hitl_active=True),
                                      unsafe_allow_html=True)
                    status.update(label="⏸ PAUSED — Human Review Required!", state="running")
                else:
                    logs.append({"ts": _ts(), "level": "INFO", "event": "node_started",
                                 "node": "generate_report"})
                    result = node_generate_report(state, logs=logs)
                    state.update(result)
                    completed.add("generate_report")
                    graph_ph.markdown(pipeline_html(completed=completed), unsafe_allow_html=True)
                    st.session_state.s6_state = state
                    st.session_state.s6_done = True
                    status.update(label="✅ Pipeline complete!", state="complete")
                    report_review(state, state.get("filename", "code.py"))
            except Exception as e:
                status.update(label=f"❌ {e}", state="error")
                st.error(str(e))

    # HITL gate
    if st.session_state.s6_hitl_needed and not st.session_state.s6_done:
        state = st.session_state.s6_state
        st.divider()
        st.error("## 🚨 AGENT PAUSED — HUMAN REVIEW REQUIRED")
        c1, c2 = st.columns(2)
        approve = c1.button("✅ APPROVE", type="primary", key="s6_approve")
        reject  = c2.button("❌ REJECT",               key="s6_reject")

        if approve or reject:
            approved = approve
            logs = st.session_state.s6_logs
            logs.append({"ts": _ts(), "level": "INFO", "event": "hitl_decision",
                         "approved": approved})
            state["human_approved"] = approved
            state["processing_steps"].append("human_review_gate")
            logs.append({"ts": _ts(), "level": "INFO", "event": "node_started",
                         "node": "generate_report"})
            result = node_generate_report(state, human_approved=approved, logs=logs)
            state.update(result)
            st.session_state.s6_state = state
            st.session_state.s6_hitl_needed = False
            st.session_state.s6_done = True
            report_review(state, state.get("filename", "code.py"))
            st.rerun()

    # Show logs and results
    if st.session_state.s6_logs:
        st.divider()
        st.markdown("#### 📡 Live Log Stream")
        st.caption("This is what Datadog / CloudWatch sees. Build alerts on any of these events.")
        log_display(st.session_state.s6_logs)

    if st.session_state.s6_done and st.session_state.s6_state:
        state = st.session_state.s6_state
        st.divider()
        st.markdown("### 📊 Results")
        col1, col2 = st.columns([1, 1])
        with col1:
            risk_bar(state["risk_score"], state["risk_label"])
        with col2:
            issue_breakdown_bar(state.get("all_issues", []))
        perf_metrics(state["token_usage"], state["elapsed_ms"])

        if st.button("🔄 Reset", key="s6_reset"):
            for k in ["s6_logs", "s6_state", "s6_hitl_needed", "s6_done"]:
                del st.session_state[k]
            st.rerun()


def show_step7():
    st.header("Step 7 — Wrap as a REST API 🚀")
    st.caption("*'From script to service.'*  |  FastAPI wraps the entire pipeline")

    st.markdown("""<div class="analogy-box">
<h3>🏥 Real World: Hospital Reception Desk + Patient Tracking System</h3>
Before: doctors wandering around, patients stopping them in hallways.<br>
After: patients <strong>call the reception desk (POST /review)</strong>,
get a ticket number, and check back later <strong>(GET /review/{id})</strong>.<br><br>
Now <strong>any system</strong> — CI/CD pipeline, GitHub Action, IDE plugin, Slack bot —
can submit code for review and poll for results.
<strong>Scales to millions of reviews. Zero manual work.</strong>
</div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### The 4 API endpoints")
        endpoints = [
            ("POST", "/review",         "Submit code → get review_id",                    "Start async pipeline"),
            ("GET",  "/review/{id}",    "Poll for results",                               "pending → running → awaiting_approval → complete"),
            ("POST", "/approve/{id}",   "Human approval via HTTP",                        "Replaces the terminal input() prompt"),
            ("GET",  "/reviews",        "List all reviews",                               "Dashboard / audit trail"),
        ]
        for method, path, what, how in endpoints:
            color = "#22c55e" if method == "GET" else "#3b82f6"
            st.markdown(f"""
<div style="background:#0f172a;border-radius:8px;padding:10px 14px;margin:6px 0;border-left:3px solid {color};">
<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:bold;">{method}</span>
<code style="color:#e2e8f0;margin-left:8px;">{path}</code><br>
<span style="color:#94a3b8;font-size:13px;">{what}</span><br>
<span style="color:#64748b;font-size:12px;">{how}</span>
</div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("#### Request / Response")
        st.code("""\
# 1. Submit code for review
POST /review
{
  "code": "...",
  "filename": "payments.py"
}
→ {"review_id": "abc-123", "status": "pending"}

# 2. Poll for results
GET /review/abc-123
→ {
    "status": "awaiting_approval",
    "risk_score": 85,
    "risk_label": "CRITICAL",
    "all_issues": [...],
  }

# 3. Human approves via HTTP
POST /approve/abc-123
{"approved": true, "reviewer": "alice"}
→ {"approved": true}""", language="json")

    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.markdown("""#### CI/CD Integration
```yaml
# .github/workflows/review.yml
- name: Security Review
  run: |
    ID=$(curl -s -X POST localhost:8000/review \\
      -d '{"code": "..."}' | jq -r .review_id)
    # Poll until complete
    # Block merge if CRITICAL found
```""")

    col2.markdown("""#### IDE Plugin
```python
# On save in VS Code:
resp = requests.post(
    "http://localhost:8000/review",
    json={"code": editor.text}
)
# Show inline annotations
# for each issue.line_number
```""")

    col3.markdown("""#### Slack Bot
```python
@app.command("/review")
def review_cmd(ack, body):
    # Submit to API
    # Post results to channel
    # Tag user if CRITICAL
    # Require approval reaction
```""")

    st.divider()
    st.markdown("#### How to run Step 7")
    st.code("python step7_serve.py", language="bash")

    st.info("""**What happens:**
1. FastAPI starts on http://localhost:8000
2. Browser opens automatically
3. API docs at http://localhost:8000/docs
4. POST code → watch nodes run in terminal logs → GET results in browser""")

    st.markdown("#### The journey we took")
    journey = [
        ("Step 1", "🤷", "8 lines. Raw blob. Ships Friday. Can't answer any questions."),
        ("Step 2", "📋", "Typed state. Every field named. Every finding structured."),
        ("Step 3", "🔧", "First node. One job. Independently testable."),
        ("Step 4", "🔒", "Security specialist. OWASP Top 10. Exact line numbers."),
        ("Step 5", "🏭", "5 nodes. Risk score 0-100. Human-in-the-Loop gate."),
        ("Step 6", "📡", "Structured logs. Build Datadog alerts. Cost tracking."),
        ("Step 7", "🚀", "REST API. CI/CD ready. Scales to millions."),
    ]
    for step, emoji, desc in journey:
        st.markdown(f"**{step} {emoji}** — {desc}")

    st.success("""**From vibe-coded assistant to production-grade agent.**
Same Claude API. Same core logic. Completely different capability.
That's the LangGraph pattern.""")


def show_step8():
    st.header("Step 8 — Docker: Package It 📦")
    st.caption("*'Works on my machine' → 'Works on every machine'*")

    st.markdown("""<div class="analogy-box">
<h3>🏥 Real World: The Sterile Surgical Kit</h3>
The surgical team doesn't carry loose scalpels in their pockets.
Every instrument is <strong>packaged in a sterile kit</strong> — sealed, standardized, and works the same
whether opened in Tokyo, Toronto, or Texas.<br><br>
Docker is that kit. Your agent, its dependencies, its config — sealed in one image.
<strong>Run it anywhere: laptop, CI server, Kubernetes cluster. Same behavior. Every time.</strong>
</div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### `Dockerfile` — 6 meaningful lines")
        st.code("""FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s \\
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "step7_serve:app", \\
     "--host", "0.0.0.0", "--port", "8000"]""", language="dockerfile")

        st.markdown("#### Build & run")
        st.code("""# Build the image
docker build -t code-review-agent .

# Run locally (same as step 7, now containerized)
docker run -p 8000:8000 \\
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \\
  code-review-agent

# → http://localhost:8000  ← identical behaviour""", language="bash")

    with col2:
        st.markdown("#### `docker-compose.yml` — for local dev")
        st.code("""version: "3.9"
services:
  agent:
    build: .
    ports:
      - "8000:8000"
    env_file: .env          # reads your .env file
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f",
             "http://localhost:8000/health"]
      interval: 30s
      retries: 3""", language="yaml")

        st.code("docker compose up --build", language="bash")

        st.divider()
        st.markdown("#### What changed from Step 7")
        st.success("""**One file added:** `Dockerfile`

Same code. Same API. Same endpoints.
No more "install Python 3.11, run pip install, set env vars..."

The image is your deployable artifact.""")

    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.info("**Local dev**\n`docker compose up`\nFull stack, one command")
    col2.info("**CI/CD**\n`docker build && push`\nImage → registry")
    col3.info("**Production**\n`docker pull && run`\nDeploy from registry")

    st.markdown("#### Why this matters for the agent")
    st.markdown("""
- Step 7's server only runs when YOU run it on YOUR laptop
- With Docker, the agent can run on **any server, any cloud, any CI runner**
- Your GitHub Action (Step 9) can call a dockerized agent running in the cloud
- Your teammate deploys with `docker compose up` — no Python setup, no dependency hell
""")


def show_step9():
    st.header("Step 9 — GitHub Actions: Auto-Review Every PR 🔄")
    st.caption("*'Every PR reviewed. No human bottleneck. Critical issues blocked.'*")

    st.markdown("""<div class="analogy-box">
<h3>🏥 Real World: The Quality Control Inspector on the Assembly Line</h3>
Every car leaving the factory passes through QC. Not randomly — <em>every single one</em>.
The inspector doesn't wait to be asked. The <strong>conveyor belt triggers it automatically</strong>.<br><br>
GitHub Actions is that conveyor belt. Every time a developer opens a PR, the agent fires automatically.
It reviews the changed files, posts findings as inline PR comments, and
<strong>blocks the merge if critical vulnerabilities are found</strong>. Zero human bottleneck.
</div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### `.github/workflows/code-review.yml`")
        st.code("""name: AI Code Review
on:
  pull_request:
    types: [opened, synchronize]
    paths: ['**.py']          # only fires when .py changes

jobs:
  review:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write   # post comments
      contents: read

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0      # need full history for diff

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - run: pip install anthropic requests

      - name: Review changed files
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GITHUB_REPOSITORY: ${{ github.repository }}
          PR_NUMBER: ${{ github.event.pull_request.number }}
        run: python scripts/review_pr.py""", language="yaml")

    with col2:
        st.markdown("#### `scripts/review_pr.py` — the logic")
        st.code("""# 1. Get changed .py files in this PR
changed = get_changed_python_files()
#    → ["payments.py", "auth.py"]

# 2. Review each with Claude
for filepath in changed:
    data = review_file(client, filepath)
    issues += data["issues"]

# 3. Post as GitHub PR comment
body = format_comment(all_issues)
post_pr_comment(body)

# 4. Block merge if critical found
if has_critical:
    sys.exit(1)   # ← fails status check""", language="python")

        st.markdown("#### The two outcomes")
        c1, c2 = st.columns(2)
        c1.success("✅ **No critical issues**\nStatus check passes\nMerge button active")
        c2.error("🚨 **Critical found**\nStatus check fails\nMerge button greyed out")

        st.markdown("#### Setup — 3 steps")
        st.code("""# 1. Add secret in GitHub UI:
#    Repo → Settings → Secrets → Actions
#    ANTHROPIC_API_KEY = sk-ant-...

# 2. Push these two files:
git add .github/workflows/code-review.yml
git add scripts/review_pr.py
git push

# 3. Open any PR with .py changes.
#    That's it. Agent reviews automatically.""", language="bash")

    st.divider()
    st.markdown("### 👁 Live Demo: What the PR Comment Looks Like")
    st.caption("Paste code below — we'll run the exact same review the GitHub Action would run and render it as a PR comment.")

    code_input = st.text_area("Code (simulating a PR diff):", value=DEFAULT_CODE, height=150, key="s9_code")

    if st.button("▶ Simulate GitHub Action Review", type="primary", key="s9_run"):
        with st.spinner("Running review (same as the GitHub Action)..."):
            try:
                t0 = time.time()
                r = client.messages.create(
                    model=MODEL,
                    max_tokens=2048,
                    system="Senior security engineer. Return ONLY valid JSON, no markdown.",
                    messages=[{"role": "user", "content": f"""Review this code for security vulnerabilities and bugs.
Return JSON: {{"issues":[{{"severity":"critical|high|medium|low","line_number":null,"title":"","description":"","suggestion":""}}],"has_critical":false}}

FILE: buggy_code.py
CODE:
{code_input}"""}],
                )
                text = r.content[0].text.strip()
                if text.startswith("```"):
                    text = "\n".join(text.split("\n")[1:-1])
                data = json.loads(text)
                elapsed = round((time.time() - t0) * 1000, 1)
                tokens = r.usage.input_tokens + r.usage.output_tokens
            except Exception as e:
                st.error(f"Error: {e}")
                return

        issues = data.get("issues", [])
        has_critical = data.get("has_critical", any(i.get("severity") == "critical" for i in issues))

        SEV_EMOJI  = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
        SEV_COLOR  = {"critical": "#f85149", "high": "#e3b341", "medium": "#f0883e", "low": "#3fb950"}
        border_col = "#f85149" if has_critical else "#3fb950"
        badge_txt  = "🚨 MERGE BLOCKED" if has_critical else "✅ PASSED"
        header_txt = "🚨 CRITICAL ISSUES FOUND — Merge Blocked" if has_critical else "🤖 AI Code Review"

        issue_html = ""
        for issue in issues:
            sev   = issue.get("severity", "low")
            col   = SEV_COLOR.get(sev, "#888")
            emoji = SEV_EMOJI.get(sev, "⚪")
            ln    = f" · line {issue['line_number']}" if issue.get("line_number") else ""
            issue_html += f"""
<div style="border-left:3px solid {col};padding:8px 14px;margin:8px 0;background:#0f172a;border-radius:0 6px 6px 0;">
  <div style="font-weight:700;color:#e6edf3;font-size:14px;">{emoji} [{sev.upper()}] {issue.get('title','')}{ln}</div>
  <div style="color:#94a3b8;font-size:13px;margin-top:4px;">{issue.get('description','')}</div>
  <div style="color:#60a5fa;font-size:13px;margin-top:6px;"><strong>Fix:</strong> {issue.get('suggestion','')}</div>
</div>"""

        st.markdown(f"""<div style="border:1px solid #30363d;border-radius:10px;overflow:hidden;margin:16px 0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
  <div style="background:#161b22;padding:10px 16px;display:flex;align-items:center;gap:10px;border-bottom:1px solid #30363d;">
    <div style="width:32px;height:32px;border-radius:50%;background:#238636;display:flex;align-items:center;justify-content:center;font-size:16px;">🤖</div>
    <span style="color:#e6edf3;font-weight:600;">code-review-agent</span>
    <span style="color:#7d8590;font-size:13px;">commented just now</span>
    <span style="margin-left:auto;padding:3px 10px;border-radius:12px;border:1px solid {border_col};color:{border_col};font-size:12px;font-weight:700;">{badge_txt}</span>
  </div>
  <div style="background:#0d1117;padding:16px;">
    <div style="color:#e6edf3;font-size:16px;font-weight:700;margin-bottom:12px;">{header_txt}</div>
    {issue_html}
    <div style="color:#7d8590;font-size:12px;margin-top:14px;border-top:1px solid #21262d;padding-top:10px;">
      Reviewed <code style="background:#21262d;padding:1px 5px;border-radius:3px;">buggy_code.py</code>
      &nbsp;·&nbsp; {len(issues)} issue(s) &nbsp;·&nbsp; {tokens:,} tokens &nbsp;·&nbsp; {elapsed:.0f}ms
      &nbsp;·&nbsp; Powered by Claude Opus 4.5
    </div>
  </div>
</div>""", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Issues found", len(issues))
        c2.metric("Tokens used", f"{tokens:,}")
        c3.metric("Status check", "❌ BLOCKED" if has_critical else "✅ PASS")

        if has_critical:
            st.error("**In a real PR:** this status check fails. The merge button is greyed out. No critical code ships without review.")
        else:
            st.success("**In a real PR:** status check passes. Developer can merge.")


def show_step10():
    st.header("Step 10 — Deploy to Production ☁️")
    st.caption("*'From localhost:8000 to https://your-agent.fly.dev'*")

    st.markdown("""<div class="analogy-box">
<h3>🏥 Real World: Opening the Hospital to the Public</h3>
You've built the hospital. Tested every specialist. Ran drills. Everything works.<br>
Now you <strong>open the doors</strong> — to every patient, every referring doctor, every ambulance.<br><br>
Deploying to the cloud means your agent is no longer tied to your laptop.
Available 24/7. Accessible from your GitHub Action, Slack bot, IDE plugin,
your teammate in a different timezone. <strong>From anywhere on the internet.</strong>
</div>""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🚀 Fly.io (Recommended)", "🚂 Railway (Even Simpler)"])

    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### `fly.toml` — one config file")
            st.code("""app = "code-review-agent"
primary_region = "sjc"

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = true   # scale to zero when idle
  auto_start_machines = true  # wake on first request

[[vm]]
  cpu_kind = "shared"
  cpus = 1
  memory_mb = 512""", language="toml")

        with col2:
            st.markdown("#### Deploy in 3 commands")
            st.code("""# 1. Install flyctl
curl -L https://fly.io/install.sh | sh

# 2. First-time launch
fly launch
# reads fly.toml
# builds Docker image
# deploys to Fly infrastructure

# 3. Set your API key secret
fly secrets set ANTHROPIC_API_KEY=sk-ant-...

# Done. Your agent is live:
# https://code-review-agent.fly.dev""", language="bash")

            st.code("""# Every deploy after that:
fly deploy   # ← one command, ~60 seconds""", language="bash")

        st.markdown("#### Update your GitHub Action to use the live URL")
        st.code("""# Option A: run script directly in CI (current — works great)
run: python scripts/review_pr.py

# Option B: call your deployed cloud agent instead
- name: Submit for review
  run: |
    ID=$(curl -s -X POST \\
      https://code-review-agent.fly.dev/review \\
      -H "Content-Type: application/json" \\
      -d '{\"code\":\"...\",\"filename\":\"payments.py\"}' \\
      | jq -r .review_id)
    # poll /review/$ID, post comment, exit 1 if critical""", language="yaml")

    with tab2:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### Railway — literally zero config")
            st.code("""# No fly.toml needed.
# Just connect your GitHub repo.

# 1. railway.app → New Project
#    → "Deploy from GitHub repo"
#    → select this repo

# 2. Add environment variable:
#    ANTHROPIC_API_KEY = sk-ant-...

# 3. Railway detects the Dockerfile,
#    builds and deploys automatically.

# Every git push = new deploy.
# Done.""", language="text")

        with col2:
            st.markdown("#### Which to choose")
            st.markdown("""
| | Railway | Fly.io |
|---|---|---|
| Setup | 2 clicks | 3 commands |
| Auto-deploy on push | ✅ | Manual |
| Free tier | $5/mo credit | Generous free |
| Scale to zero | ✅ | ✅ |
| Custom domains | ✅ | ✅ |
| Best for | Speed | Control |
""")
            st.info("For this webinar demo: **Railway** is fastest to show live.\nFor real teams: **Fly.io** gives more control.")

    st.divider()
    st.markdown("### The complete journey: vibe-coding → production agent")

    journey = [
        ("Step 1",  "🤷", "8 lines. Raw text blob. Ships Friday. Can't answer any questions."),
        ("Step 2",  "📋", "TypedDict state. Every field named and typed."),
        ("Step 3",  "🔧", "First LangGraph node. One job. Independently testable."),
        ("Step 4",  "🔒", "Security specialist node. OWASP Top 10. Exact line numbers."),
        ("Step 5",  "🏭", "Full pipeline. Risk score 0–100. Human-in-the-Loop gate."),
        ("Step 6",  "📡", "Structured JSON logs. Build Datadog/CloudWatch alerts."),
        ("Step 7",  "🚀", "FastAPI REST API. Any system can call it."),
        ("Step 8",  "📦", "Docker. Runs anywhere. Your deployable artifact."),
        ("Step 9",  "🔄", "GitHub Actions. Every PR reviewed. Critical = blocked."),
        ("Step 10", "☁️", "Cloud deploy. Available to your entire team, 24/7."),
    ]
    for step, emoji, desc in journey:
        st.markdown(f"**{step} {emoji}** — {desc}")

    st.success("""**That's the complete arc: from vibe-coding to a production-grade AI agent.**

Same Claude API throughout. 10 steps. Each adds exactly one concept.
The 8-line function you wrote in Step 1 is the same intelligence running in production.
The difference is everything around it — structure, pipeline, observability, packaging, automation, deployment.""")

    st.divider()
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### What you built")
        st.markdown("""
- Structured JSON output with typed state
- Multi-node pipeline with specialist agents
- Risk scoring (0–100) with OWASP references
- Human-in-the-Loop approval gate
- Structured logs (Datadog/CloudWatch-ready)
- REST API (POST /review, GET /review/{id}, POST /approve/{id})
- Docker container (runs anywhere)
- GitHub Action (auto-reviews every PR, blocks on critical)
- Cloud deployment (Fly.io / Railway)
""")
    with col2:
        st.markdown("#### What's next (for your real project)")
        st.markdown("""
- **Database** — PostgreSQL to persist review history
- **Auth** — API key middleware for multi-team access
- **Queue** — Redis + Celery for high-volume async processing
- **More nodes** — performance analysis, license scanning, test coverage
- **Slack/Teams bot** — post findings to channels
- **IDE plugin** — inline annotations in VS Code
- **Dashboard** — track trends, cost, team metrics over time
""")


# ═════════════════════════════════════════════════════════════════════
#  SIDEBAR + ROUTER
# ═════════════════════════════════════════════════════════════════════

STEPS = {
    1:  ("🤷", "Vibe Coded",            "It works. Ship it."),
    2:  ("📋", "Typed State",           "Give the agent memory."),
    3:  ("🔧", "First Node",            "One node. One job."),
    4:  ("🔒", "Security Node",         "A specialist joins."),
    5:  ("🏭", "Full Pipeline + HITL",  "5 nodes. Human gate."),
    6:  ("📡", "Observability",         "CCTV for your agent."),
    7:  ("🚀", "REST API",              "Script → Service."),
    8:  ("📦", "Docker",               "Package it. Runs anywhere."),
    9:  ("🔄", "GitHub Actions",        "Auto-review every PR."),
    10: ("☁️", "Deploy to Cloud",       "localhost → internet."),
}

with st.sidebar:
    st.markdown("## 🤖 LangGraph Agent\n### Webinar Live Demo")
    st.caption("*Vibe Coding → Production Agent*")
    st.divider()

    step = st.radio(
        "Select Step",
        options=list(STEPS.keys()),
        format_func=lambda x: f"Step {x}: {STEPS[x][1]}",
        index=0,
        key="nav_step",
    )

    emoji, title, tagline = STEPS[step]
    st.divider()
    st.markdown(f"**{emoji} {title}**\n\n_{tagline}_")
    st.divider()

    st.markdown("""**Each step adds one concept:**
1. Raw text output
2. Typed state object
3. LangGraph node pattern
4. Specialist node
5. Full graph + HITL
6. Structured logging
7. Production REST API
8. Docker packaging
9. GitHub Actions CI/CD
10. Cloud deployment

Built on Claude Opus 4.5""")

    st.divider()
    if step > 1:
        if st.button("← Previous step", key="nav_prev"):
            st.session_state.nav_step = step - 1
            st.rerun()
    if step < 10:
        if st.button("Next step →", key="nav_next"):
            st.session_state.nav_step = step + 1
            st.rerun()

# ── Route to step renderer ────────────────────────────────────────────
renderers = {
    1:  show_step1,
    2:  show_step2,
    3:  show_step3,
    4:  show_step4,
    5:  show_step5,
    6:  show_step6,
    7:  show_step7,
    8:  show_step8,
    9:  show_step9,
    10: show_step10,
}
renderers[step]()
