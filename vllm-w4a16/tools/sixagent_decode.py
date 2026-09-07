import json,time,threading,urllib.request,statistics,sys
sys.argv=[sys.argv[0]]; exec(open("/private/tmp/claude-501/-Users-clawdbot/ef3eadcb-21f6-4df8-b0d8-0fccf98884bb/scratchpad/sixagent_load.py").read().split("results=[]; lock=threading.Lock()")[0])
URL="http://100.113.64.18:8090/v1/chat/completions"; N=6; ROUNDS=2
results=[]; lock=threading.Lock()
def one(agent):
    for rnd in range(ROUNDS):
        msgs=[{"role":"system","content":SYSTEM},{"role":"user","content":TASKS[(agent+rnd)%6]+" Before touching any tool, write the complete implementation plan as prose: every file you will read and change, the exact code you intend to write for the core function, the tests you will add, and the risks. At least 600 words. Do not call tools in this message."}]
        body=json.dumps({"model":"qwen3.8-flash-next","messages":msgs,"tools":TOOLS,"tool_choice":"none","max_tokens":700,"temperature":0.2,"stream":True,"stream_options":{"include_usage":True}}).encode()
        req=urllib.request.Request(URL,data=body,headers={"Content-Type":"application/json"})
        t0=time.time(); first=None; last=t0; n=0; pt=0; chunks=0
        with urllib.request.urlopen(req,timeout=600) as r:
            for line in r:
                line=line.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"): continue
                d=json.loads(line[5:])
                if d.get("choices") and (d["choices"][0]["delta"].get("content") or ""):
                    if first is None: first=time.time()
                    last=time.time(); chunks+=1
                if d.get("usage"): n=d["usage"]["completion_tokens"]; pt=d["usage"]["prompt_tokens"]
        with lock: results.append({"agent":agent,"round":rnd,"prompt_tokens":pt,"completion_tokens":n,"chunks":chunks,"ttft":first-t0,"decode_tps":n/(last-first),"e2e":last-t0})
T0=time.time(); th=[threading.Thread(target=one,args=(a,)) for a in range(N)]
[t.start() for t in th]; [t.join() for t in th]; wall=time.time()-T0
tt=sorted(r["ttft"] for r in results); p=lambda q: tt[min(len(tt)-1,int(round(q*(len(tt)-1))))]
tot=sum(r["completion_tokens"] for r in results)
summary={"agents":N,"rounds":ROUNDS,"requests":len(results),"wall_s":round(wall,1),"prompt_tokens":{"min":min(r["prompt_tokens"] for r in results),"max":max(r["prompt_tokens"] for r in results)},
 "completion_tokens":{"min":min(r["completion_tokens"] for r in results),"median":int(statistics.median(r["completion_tokens"] for r in results)),"total":tot},
 "aggregate_output_tps_wall":round(tot/wall,1),"ttft_s":{"p50":round(p(0.5),2),"p90":round(p(0.9),2),"p99":round(p(0.99),2)},
 "per_stream_decode_tps":{"min":round(min(r["decode_tps"] for r in results),1),"median":round(statistics.median(r["decode_tps"] for r in results),1),"max":round(max(r["decode_tps"] for r in results),1)}}
print(json.dumps(summary)); json.dump({"summary":summary,"requests":results},open("/private/tmp/claude-501/-Users-clawdbot/ef3eadcb-21f6-4df8-b0d8-0fccf98884bb/scratchpad/sixagent_boot19_decode.json","w"),indent=1)
