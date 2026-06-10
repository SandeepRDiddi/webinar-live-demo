"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 7 — WRAP AS A REST API
 "From script to service."
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 What's new:
   ✦ FastAPI wraps the entire pipeline
   ✦ POST /review  — submit code, get review_id
   ✦ GET  /review/{id} — poll for structured results
   ✦ POST /approve/{id} — human approval via HTTP
   ✦ Opens browser automatically

 Run: python step7_serve.py

 DEMO SEQUENCE:
   1. Run this file
   2. Browser opens to http://localhost:8000
   3. Paste the buggy code (or load the sample)
   4. Click "Review Code"
   5. Watch nodes run in terminal logs
   6. Results appear in browser with color-coded severity
   7. If CRITICAL: approval banner appears in UI

 This is the FINAL STATE — production agent, full UI.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json, time, uuid, os, sys, asyncio, webbrowser, threading
import anthropic
import logging
from typing import TypedDict, List, Optional
from datetime import datetime, timezone
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as PydanticModel
from dotenv import load_dotenv

load_dotenv()
from agent_prism import make_client, report_review, MODEL
client = make_client()

logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)
log = logging.getLogger("agent")

def emit(level, event, **kw):
    log.info(json.dumps({"ts":datetime.now(timezone.utc).strftime("%H:%M:%S"),"level":level,"event":event,**kw}))


# ── State ───────────────────────────────────────────────────────────
class ReviewState(TypedDict):
    code:str; filename:str; language:str; line_count:int; functions:List[str]
    security_issues:List[dict]; bug_issues:List[dict]; quality_issues:List[dict]
    all_issues:List[dict]; risk_score:int; risk_label:str; final_report:str
    needs_human_review:bool; human_approved:Optional[bool]
    token_usage:dict; processing_steps:List[str]; elapsed_ms:dict
    error:Optional[str]; retry_count:int


def _call_claude(system, user, node):
    for attempt in range(1,4):
        try:
            t0=time.time()
            r=client.messages.create(model=MODEL,max_tokens=2048,system=system,messages=[{"role":"user","content":user}])
            text=r.content[0].text.strip()
            if text.startswith("```"): text="\n".join(text.split("\n")[1:-1])
            return json.loads(text), r.usage.input_tokens+r.usage.output_tokens, round((time.time()-t0)*1000,1)
        except Exception as e:
            if attempt==3: raise
            emit("WARN","retry",node=node,attempt=attempt,reason=str(e)); time.sleep(2**attempt)


def parse_code(state):
    emit("INFO","node_started",node="parse_code")
    d,tok,ms=_call_claude("Return ONLY valid JSON.",f'Return {{"language":"<l>","line_count":<n>,"functions":["..."]}}\nCODE:\n{state["code"]}',"parse_code")
    emit("INFO","node_completed",node="parse_code",ms=ms,tokens=tok)
    return {"language":d.get("language","unknown"),"line_count":d.get("line_count",0),"functions":d.get("functions",[]),
            "processing_steps":state.get("processing_steps",[])+["parse_code"],
            "token_usage":{**state.get("token_usage",{}),"parse_code":tok},"elapsed_ms":{**state.get("elapsed_ms",{}),"parse_code":ms}}

def security_scan(state):
    emit("INFO","node_started",node="security_scan")
    d,tok,ms=_call_claude("Senior security engineer. Return ONLY valid JSON.",
        f'Find ALL security vulnerabilities. Return JSON: {{"issues":[{{"severity":"critical|high|medium|low","line_number":<n>,"title":"","description":"","suggestion":"","code_fix":null,"owasp":""}}],"has_critical":false}}\nCODE:\n{state["code"]}',"security_scan")
    issues=d.get("issues",[])
    emit("INFO","node_completed",node="security_scan",ms=ms,tokens=tok,issues=len(issues))
    return {"security_issues":issues,"processing_steps":state.get("processing_steps",[])+["security_scan"],
            "token_usage":{**state.get("token_usage",{}),"security_scan":tok},"elapsed_ms":{**state.get("elapsed_ms",{}),"security_scan":ms}}

def bug_detection(state):
    emit("INFO","node_started",node="bug_detection")
    d,tok,ms=_call_claude("Senior engineer. Return ONLY valid JSON.",
        f'Find runtime bugs. Return JSON: {{"issues":[{{"severity":"critical|high|medium|low","line_number":<n>,"title":"","description":"","suggestion":"","code_fix":null}}]}}\nCODE:\n{state["code"]}',"bug_detection")
    issues=d.get("issues",[])
    emit("INFO","node_completed",node="bug_detection",ms=ms,tokens=tok,issues=len(issues))
    return {"bug_issues":issues,"processing_steps":state.get("processing_steps",[])+["bug_detection"],
            "token_usage":{**state.get("token_usage",{}),"bug_detection":tok},"elapsed_ms":{**state.get("elapsed_ms",{}),"bug_detection":ms}}

def aggregate_issues(state):
    all_i=state.get("security_issues",[])+state.get("bug_issues",[])+state.get("quality_issues",[])
    counts={}
    for i in all_i: s=i.get("severity","low"); counts[s]=counts.get(s,0)+1
    score=min(100,counts.get("critical",0)*25+counts.get("high",0)*10+counts.get("medium",0)*5+counts.get("low",0))
    label="CRITICAL" if score>=75 else "HIGH" if score>=50 else "MEDIUM" if score>=25 else "LOW"
    needs=counts.get("critical",0)>0
    if needs: emit("WARN","human_review_required",file=state["filename"],risk=score,critical=counts.get("critical",0))
    emit("INFO","node_completed",node="aggregate",risk=score,label=label,total=len(all_i))
    return {"all_issues":all_i,"risk_score":score,"risk_label":label,"needs_human_review":needs,
            "processing_steps":state.get("processing_steps",[])+["aggregate_issues"]}

def should_escalate(state): return "escalate" if state.get("needs_human_review") else "report"

def human_review_gate(state):
    emit("WARN","hitl_paused",file=state["filename"],risk=state["risk_score"])
    return {"processing_steps":state.get("processing_steps",[])+["human_review_gate"]}

def generate_report(state):
    all_i=state.get("all_issues",[])
    counts={}
    for i in all_i: s=i.get("severity","low"); counts[s]=counts.get(s,0)+1
    so={"critical":0,"high":1,"medium":2,"low":3}
    sorted_i=sorted(all_i,key=lambda x:so.get(x.get("severity","low"),4))
    report=(f"# Review: {state['filename']}\nRisk: {state['risk_score']}/100 — {state['risk_label']}\n"
            f"Issues: {len(all_i)} ({counts.get('critical',0)} critical, {counts.get('high',0)} high)\n\n")
    for i,issue in enumerate(sorted_i,1):
        ln=f" (line {issue['line_number']})" if issue.get("line_number") else ""
        report+=f"[{i}] [{issue.get('severity','').upper()}] {issue.get('title','')}{ln}\n     {issue.get('suggestion','')}\n\n"
    total_tok=sum(state.get("token_usage",{}).values())
    total_ms=sum(state.get("elapsed_ms",{}).values())
    emit("INFO","review_complete",file=state["filename"],risk=state["risk_score"],issues=len(all_i),ms=round(total_ms,1),tokens=total_tok)
    return {"final_report":report,"processing_steps":state.get("processing_steps",[])+["generate_report"]}


def build_graph():
    wf=StateGraph(ReviewState)
    for n,f in [("parse_code",parse_code),("security_scan",security_scan),("bug_detection",bug_detection),
                ("aggregate_issues",aggregate_issues),("human_review_gate",human_review_gate),("generate_report",generate_report)]:
        wf.add_node(n,f)
    wf.set_entry_point("parse_code")
    wf.add_edge("parse_code","security_scan"); wf.add_edge("security_scan","bug_detection")
    wf.add_edge("bug_detection","aggregate_issues")
    wf.add_conditional_edges("aggregate_issues",should_escalate,{"escalate":"human_review_gate","report":"generate_report"})
    wf.add_edge("human_review_gate","generate_report"); wf.add_edge("generate_report",END)
    return wf.compile(checkpointer=MemorySaver())


# ── FastAPI ─────────────────────────────────────────────────────────
app = FastAPI(title="Code Review Agent")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
reviews: dict = {}

class ReviewRequest(PydanticModel):
    code: str; filename: str = "code.py"

class ApprovalRequest(PydanticModel):
    approved: bool; reviewer: str = "human"

async def _run(review_id, code, filename):
    reviews[review_id]["status"] = "running"
    try:
        graph = build_graph()
        initial: ReviewState = {
            "code":code,"filename":filename,"language":"","line_count":0,"functions":[],
            "security_issues":[],"bug_issues":[],"quality_issues":[],"all_issues":[],
            "risk_score":0,"risk_label":"","final_report":"","needs_human_review":False,"human_approved":None,
            "token_usage":{},"processing_steps":[],"elapsed_ms":{},"error":None,"retry_count":0
        }
        loop=asyncio.get_event_loop()
        result=await loop.run_in_executor(None, lambda: graph.invoke(initial,config={"configurable":{"thread_id":review_id}}))
        reviews[review_id].update({
            "status": "awaiting_approval" if result.get("needs_human_review") else "complete",
            "risk_score": result.get("risk_score"),
            "risk_label": result.get("risk_label"),
            "all_issues": result.get("all_issues",[]),
            "final_report": result.get("final_report",""),
            "token_usage": result.get("token_usage",{}),
            "elapsed_ms": result.get("elapsed_ms",{}),
        })
        report_review(result, filename)
    except Exception as e:
        reviews[review_id].update({"status":"error","error":str(e)})

@app.get("/",response_class=HTMLResponse)
async def ui():
    return open(os.path.join(os.path.dirname(__file__),"..","code-review-agent","ui","index.html")).read()

@app.get("/health")
async def health(): return {"status":"ok"}

@app.post("/review")
async def submit(req: ReviewRequest, bg: BackgroundTasks):
    rid=str(uuid.uuid4())
    reviews[rid]={"review_id":rid,"status":"pending","filename":req.filename}
    bg.add_task(_run, rid, req.code, req.filename)
    return reviews[rid]

@app.get("/review/{rid}")
async def get_review(rid: str):
    if rid not in reviews: raise HTTPException(404)
    return reviews[rid]

@app.post("/approve/{rid}")
async def approve(rid: str, req: ApprovalRequest):
    if rid not in reviews: raise HTTPException(404)
    reviews[rid].update({"status":"complete","human_approved":req.approved,"reviewer":req.reviewer})
    return {"approved":req.approved}

@app.get("/reviews")
async def list_reviews(): return list(reviews.values())


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*55)
    print("  Code Review Agent — http://localhost:8000")
    print("  API docs          — http://localhost:8000/docs")
    print("="*55 + "\n")
    threading.Timer(1.5, lambda: webbrowser.open("http://localhost:8000")).start()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
