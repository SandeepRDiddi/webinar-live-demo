# Presenter Guide — From Vibe Coding to Production AI Agents
## Live Demo Script (70 minutes)

---

## Before You Start (5 min before webinar)

```bash
cd webinar-live-demo
source venv/bin/activate   # or venv\Scripts\activate on Windows
```

Open in VS Code with split view:
- **Left panel**: the step file you're currently on
- **Right panel**: terminal

Have `samples/buggy_code.py` open in a third tab so you can scroll through it.

---

## DEMO 1 — The Vibe Coded Version (10–30 min, ~20 min)

### What to say before running
> "Everyone in this room has written this code — or something like it.
> You get the idea, you open Cursor, you vibe your way to a working prototype.
> Let me show you what that looks like for a code reviewer."

### Open step1_vibe.py — talk through it BEFORE running
Point to each line:
- `client.messages.create(...)` — "One API call."
- `f"Review this code"` — "Completely unstructured prompt."
- `return response.content[0].text` — "Returns whatever the model felt like saying."

> "8 lines. Ships on Friday. Let's run it."

### Run it
```bash
python step1_vibe.py
```

### While it runs, scroll through `samples/buggy_code.py`
> "This is our test file. It has 10 intentional bugs — SQL injection,
> hardcoded Stripe key, plaintext passwords. Let's see what the vibe agent finds."

### When output appears — point to the screen
> "It works! It found things. But now try to answer these questions…"

Read the questions printed at the bottom:
- "How many CRITICAL issues? You'd have to count manually."
- "Which exact line has the SQL injection? Scroll and squint."
- "What's the risk score? There isn't one."
- "Can you sort by severity? No — it's a paragraph."

> "This is the gap. Not whether it finds issues — it does.
> The gap is whether you can DO anything with the output at scale."

---

## DEMO 2 — The Foundation: Typed State (30–35 min, ~5 min)

> "Before we write a single LangGraph node, we need to answer:
> what does the data look like at every step?
> In production, you don't pass strings around. You pass typed objects."

### Open step2_state.py — walk through it
Point to `ReviewState` TypedDict:
> "This is the single object that flows through every node.
> Every node reads from it. Every node writes back to it.
> LangGraph checkpoints this at every step — so if node 3 fails,
> we resume from node 3, not from scratch."

Point to `CodeIssue` model:
> "And every finding is a Pydantic model. Severity is an enum.
> Line number is an int. Code fix is a string. Not a blob — data."

### Run it
```bash
python step2_state.py
```

Point to the JSON output:
> "That's what a finding looks like. You can filter it, sort it,
> send it to Slack, store it in Postgres. The vibe version gave you a paragraph."

---

## DEMO 3 — First Node: parse_code (35–45 min, ~10 min)

> "Now let's write our first LangGraph node.
> A node is just a function. It takes state in, returns a dict of updates.
> LangGraph merges that dict into the state automatically."

### Open step3_first_node.py — walk through it
Point to the function signature:
```python
def parse_code(state: ReviewState) -> dict:
```
> "Takes state. Returns a dict. That's it. One job."

Point to `build_graph()`:
> "We register the node, set it as the entry point, wire it to END.
> One node. Already resumable. Already checkpointed. Already testable alone."

### Run it
```bash
python step3_first_node.py
```

Point to output:
> "Language detected. Line count. Function names. All typed.
> And look — tokens used, time taken. We track this from node 1.
> Now let's add the security scanner."

---

## DEMO 4 — Security Scanner Node (45–55 min, ~10 min)

> "The security scanner is the same pattern — one function, one job.
> But its prompt is specialist. It knows OWASP. It returns structured JSON."

### Open step4_security_node.py — show the DIFF from step3
Point to the security_scan function and the new prompt:
> "Notice the prompt: we're not asking 'review this code.'
> We're asking for specific JSON with specific fields.
> That's how you get consistent, parseable output."

Point to the graph changes (2 new lines):
```python
workflow.add_node("security_scan", security_scan)
workflow.add_edge("parse_code", "security_scan")
```
> "Two lines to add a node. That's the LangGraph pattern."

### Run it
```bash
python step4_security_node.py
```

### When output appears — walk through a finding
> "Look at this. Severity: CRITICAL. Line 10. Title: SQL Injection.
> OWASP reference. And a code fix — the actual parameterized query.
> The vibe version said 'you might have an injection issue.'
> This tells you exactly where, what to change, and shows you the fix."

---

## DEMO 5 — Full Pipeline + Human-in-the-Loop (55–70 min, ~15 min)

> "Now the moment this whole webinar has been building to.
> We add the remaining nodes, wire them together,
> and add the conditional edge — the HITL gate."

### Open step5_full_pipeline.py — walk through should_escalate()
```python
def should_escalate(state: ReviewState) -> str:
    return "escalate" if state.get("needs_human_review") else "report"
```
> "This is a routing function. Returns a string.
> LangGraph uses that string to decide which node runs next.
> If we found CRITICAL issues — escalate. Otherwise — go straight to report."

Point to `add_conditional_edges`:
> "This single call implements the Human-in-the-Loop pattern.
> The graph doesn't just run linearly. It makes a decision."

### Run it — THIS IS THE CLIMAX
```bash
python step5_full_pipeline.py
```

### What happens
1. Watch nodes run: `[1/5] parse_code...` → `[2/5] security_scan...` etc.
2. When it hits `aggregate_issues` and finds CRITICAL issues:
   ```
   🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
   HUMAN REVIEW REQUIRED
   🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
   ```

### At the HITL prompt — PAUSE and talk to audience
> "The graph has stopped. It's waiting for me.
> This is the answer to the question I asked at the beginning:
> 'Who owns it when the agent fails at 2am?'
> The answer is: the agent doesn't decide alone on critical findings.
> A human does. And LangGraph makes that a one-line routing function."

Type `y` to approve, show the final report.

> "Risk score 95/100. 10 findings. Sorted by severity. With fixes.
> All typed. All filterable. This is what production looks like."

---

## DEMO 6 — Observability + Web UI (Optional, if time allows)

### Run step6 to show structured logs
```bash
python step6_observability.py
```
> "Same pipeline. One new thing: JSON logs emit as each node runs.
> Every event has a timestamp, node name, elapsed time, tokens used.
> Drop these in Datadog — you get dashboards, alerts, cost tracking."

### Run step7 to show the web UI
```bash
python step7_serve.py
```
Browser opens at http://localhost:8000

> "And this is the production experience.
> Submit code through a UI. Watch findings appear with colour-coded severity.
> CRITICAL issues surface an approval banner in the browser.
> From 8 lines of vibe code to a deployable service — that's the journey."

---

## Timing Reference

| Step | File | Time | What the audience sees change |
|------|------|------|-------------------------------|
| Vibe code | step1 | 10 min | Raw text blob |
| Typed state | step2 | 5 min | Structured JSON, Pydantic model |
| First node | step3 | 10 min | 1-node LangGraph, structured parse |
| Security node | step4 | 10 min | Findings with line numbers + OWASP |
| Full pipeline + HITL | step5 | 15 min | Graph pauses, waits for human |
| Observability | step6 | 5 min | JSON log stream |
| Web UI | step7 | 5 min | Live browser UI |

---

## If Something Goes Wrong

**API call takes too long** → Talk through the code while it runs. It's fine — this is real.

**JSON parse error** → The retry logic handles it silently. If it fails after 3 attempts, say: "This is exactly why we have retry logic — transient failures happen."

**Rate limit hit** → Same as above. Backoff kicks in automatically.

**Terminal shows no output** → Check `.env` file has `ANTHROPIC_API_KEY` set.
