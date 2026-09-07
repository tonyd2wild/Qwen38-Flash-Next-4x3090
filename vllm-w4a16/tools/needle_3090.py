import json,sys,time,urllib.request,random
URL="http://100.113.64.18:8090/v1/chat/completions"
sizes=[int(x) for x in (sys.argv[1:] or ["7000","28000","53000","100000"])]
random.seed(7)
words=("the fleet ran a quiet audit of every relay and lane while the sparks idled under the aurora of a late summer sky "
       "and the ledger filled with rows about pools tokens seats and the slow patient work of measuring instead of guessing ").split()
def filler(n_tokens):
    # ~0.75 tokens per word for this prose, so words = tokens/0.75
    n=int(n_tokens/0.75); out=[]
    while len(out)<n: out.extend(random.sample(words,len(words)))
    return " ".join(out[:n])
NEEDLE="The secret code for the 3090 lane is 4471-ROSEWOOD."
for tk in sizes:
    body_text=filler(tk); cut=int(len(body_text)*0.4)
    prompt=body_text[:cut]+" "+NEEDLE+" "+body_text[cut:]+"\n\nQuestion: What is the secret code for the 3090 lane? Answer with the code only."
    req=urllib.request.Request(URL,data=json.dumps({"model":"qwen3.8-flash-next","messages":[{"role":"user","content":prompt}],"max_tokens":30,"temperature":0,"stream":True,"stream_options":{"include_usage":True}}).encode(),headers={"Content-Type":"application/json"})
    t0=time.time(); first=None; txt=""; pt=0
    try:
        with urllib.request.urlopen(req,timeout=900) as r:
            for line in r:
                line=line.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"): continue
                d=json.loads(line[5:])
                if d.get("choices"):
                    c=d["choices"][0]["delta"].get("content") or ""
                    if c and first is None: first=time.time()
                    txt+=c
                if d.get("usage"): pt=d["usage"]["prompt_tokens"]
        ttft=(first or time.time())-t0
        ok="4471-ROSEWOOD" in txt
        print(f"needle {tk:>6} target -> {pt} prompt tok, TTFT {ttft:.1f}s ({pt/ttft:,.0f} tok/s prefill), {'CORRECT' if ok else 'WRONG: '+txt.strip()[:80]}",flush=True)
    except Exception as e:
        print(f"needle {tk}: ERROR {str(e)[:200]}",flush=True)
