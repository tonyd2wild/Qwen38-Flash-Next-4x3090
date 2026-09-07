import json,time,threading,urllib.request,statistics,sys
URL="http://100.113.64.18:8090/v1/chat/completions"; N=6; TURNS=3
TOOLS=[{"type":"function","function":{"name":n,"description":d,"parameters":{"type":"object","properties":{"path":{"type":"string","description":"absolute path"},"content":{"type":"string","description":"file content or command text"},"pattern":{"type":"string","description":"regex"},"line":{"type":"integer"}},"required":["path"]}}} for n,d in [
 ("read_file","Read a file from the repository and return its contents with line numbers."),
 ("write_file","Write the given content to a file, creating parent directories as needed."),
 ("edit_file","Replace an exact string in a file with new content; fails if the string is not unique."),
 ("run_command","Run a shell command in the project root and return stdout, stderr and the exit code."),
 ("search","Search the repository for a regex pattern and return matching file paths and lines."),
 ("list_dir","List files and directories under a path, one level deep."),
 ("git_diff","Return the current git diff for the working tree or a given path."),
 ("run_tests","Run the project's test suite, optionally filtered by a pattern, and return the summary.")]]
SYSTEM=("You are a coding agent working inside a Python service repository (FastAPI + SQLAlchemy + pytest). Use the tools to inspect and modify files. "
 "Always read a file before editing it. Keep edits minimal. After changes, run the tests. Explain each step briefly, then call the next tool. "
 "Repository layout: app/main.py (routes), app/models.py (ORM), app/services/orders.py (business logic), app/db.py, tests/ (pytest), scripts/, README.md. "
 "Coding standards: type hints everywhere, no bare excepts, log with structlog, keep functions under 40 lines, docstrings in Google style. "
 "Never delete files. Never run destructive git commands. If a tool fails, read the error and try a different approach once before asking.")
TASKS=["Add an idempotency key to POST /orders so repeated requests with the same key return the original order. Plan the change, then start with the models.",
 "The nightly job in scripts/reconcile.py double counts refunds when an order has partial shipments. Find the bug and propose a fix with a test.",
 "Add rate limiting to the public /search endpoint using a token bucket per API key, configurable via settings. Start by reading app/main.py.",
 "Migrate app/services/orders.py from synchronous SQLAlchemy sessions to async sessions without changing the public function signatures.",
 "Write a CLI in scripts/export_orders.py that streams orders as CSV for a date range, with pagination and a progress bar. Begin with the data model.",
 "Our tests take 9 minutes. Profile tests/ and propose three concrete changes to cut that in half, then implement the first one."]
FAKE_TOOL_RESULT=("read_file result:\n"+"\n".join(f"{i:4d}  " + l for i,l in enumerate([
 "from fastapi import FastAPI, Depends, HTTPException","from sqlalchemy.orm import Session","from app.db import get_session","from app.models import Order, OrderItem, Refund","from app.services import orders as order_service","import structlog","","log = structlog.get_logger()","app = FastAPI(title='orders')","",
 "@app.post('/orders')","def create_order(payload: dict, session: Session = Depends(get_session)):","    try:","        order = order_service.create(session, payload)","    except ValueError as exc:","        raise HTTPException(status_code=400, detail=str(exc))","    log.info('order.created', order_id=order.id)","    return order.to_dict()","",
 "@app.get('/orders/{order_id}')","def get_order(order_id: int, session: Session = Depends(get_session)):","    order = session.get(Order, order_id)","    if order is None:","        raise HTTPException(status_code=404)","    return order.to_dict()","",
 "@app.get('/search')","def search(q: str, limit: int = 20, session: Session = Depends(get_session)):","    return [o.to_dict() for o in order_service.search(session, q, limit)]"]*3,1)))
results=[]; lock=threading.Lock()
def one(agent):
    msgs=[{"role":"system","content":SYSTEM},{"role":"user","content":TASKS[agent]}]
    for turn in range(TURNS):
        body=json.dumps({"model":"qwen3.8-flash-next","messages":msgs,"tools":TOOLS,"tool_choice":"none","max_tokens":500,"temperature":0.2,"stream":True,"stream_options":{"include_usage":True}}).encode()
        req=urllib.request.Request(URL,data=body,headers={"Content-Type":"application/json"})
        t0=time.time(); first=None; last=t0; n=0; pt=0; txt=""
        try:
            with urllib.request.urlopen(req,timeout=600) as r:
                for line in r:
                    line=line.decode().strip()
                    if not line.startswith("data:") or line.endswith("[DONE]"): continue
                    d=json.loads(line[5:])
                    if d.get("choices"):
                        dl=d["choices"][0]["delta"]; c=dl.get("content") or ""
                        tc=dl.get("tool_calls")
                        if c or tc:
                            if first is None: first=time.time()
                            last=time.time(); txt+=c
                    if d.get("usage"): n=d["usage"]["completion_tokens"]; pt=d["usage"]["prompt_tokens"]
            rec={"agent":agent,"turn":turn,"prompt_tokens":pt,"completion_tokens":n,"ttft":(first or last)-t0,"decode_tps":n/max(last-(first or t0),1e-6),"e2e":last-t0,"error":None}
        except Exception as e:
            rec={"agent":agent,"turn":turn,"prompt_tokens":pt,"completion_tokens":n,"ttft":None,"decode_tps":None,"e2e":time.time()-t0,"error":str(e)[:120]}
        with lock: results.append(rec)
        msgs.append({"role":"assistant","content":txt or "(tool call issued)"})
        msgs.append({"role":"user","content":FAKE_TOOL_RESULT+"\n\nContinue with the next step."})
T0=time.time(); th=[threading.Thread(target=one,args=(a,)) for a in range(N)]
[t.start() for t in th]; [t.join() for t in th]; wall=time.time()-T0
ok=[r for r in results if r["error"] is None]; tt=[r["ttft"] for r in ok]; tt.sort()
tot_out=sum(r["completion_tokens"] for r in ok); tot_in=sum(r["prompt_tokens"] for r in ok)
p=lambda q: tt[min(len(tt)-1,int(round(q*(len(tt)-1))))]
summary={"agents":N,"turns":TURNS,"requests":len(results),"errors":len(results)-len(ok),"wall_s":round(wall,1),
 "prompt_tokens":{"min":min(r["prompt_tokens"] for r in ok),"median":int(statistics.median(r["prompt_tokens"] for r in ok)),"max":max(r["prompt_tokens"] for r in ok),"total":tot_in},
 "completion_tokens_total":tot_out,"aggregate_output_tps_wall":round(tot_out/wall,1),
 "ttft_s":{"p50":round(p(0.5),2),"p90":round(p(0.9),2),"p99":round(p(0.99),2),"max":round(tt[-1],2)},
 "per_stream_decode_tps":{"min":round(min(r["decode_tps"] for r in ok),1),"median":round(statistics.median(r["decode_tps"] for r in ok),1),"max":round(max(r["decode_tps"] for r in ok),1)}}
print(json.dumps(summary,indent=1)); print("--- per request"); 
for r in sorted(results,key=lambda r:(r["agent"],r["turn"])): print(r)
json.dump({"summary":summary,"requests":results},open(sys.argv[1] if len(sys.argv)>1 else "sixagent_load.json","w"),indent=1)
