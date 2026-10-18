import json,os,re,sys,time,urllib.request,random
# Prefix-cache check for the PREFIX_CACHE=1 lane: (1) the same ~50K-token prompt twice (cold, then warm: TTFT must
# drop and the output must match at temperature 0), (2) a 3-turn conversation where each turn resends the last one plus
# a new question, the way agent clients do. Hits are read from the server's /metrics counters around each request.
# Usage: URL=http://localhost:8090 python3 prefix_cache_3090.py [prompt_tokens] [tokens_added_per_turn]
BASE=os.environ.get("URL","http://100.113.64.18:8090").rstrip("/"); MODEL=os.environ.get("MODEL","qwen3.8-flash-next")
TOKENS=int(sys.argv[1]) if len(sys.argv)>1 else 50000
TURN_TOKENS=int(sys.argv[2]) if len(sys.argv)>2 else 3000
salt="%08x"%random.SystemRandom().getrandbits(32)  # fresh prefix each run, so nothing is warm from an earlier run
random.seed(salt)
words=("the fleet ran a quiet audit of every relay and lane while the sparks idled under the aurora of a late summer sky "
       "and the ledger filled with rows about pools tokens seats and the slow patient work of measuring instead of guessing ").split()
def filler(n_tokens):
    n=int(n_tokens/0.75); out=[]
    while len(out)<n: out.extend(random.sample(words,len(words)))
    return " ".join(out[:n])
FACTS=" Run %s notes: the relay on bay seven is named Juniper. The spare lane token is 9182. The audit closes on a Thursday. "%salt
def metrics():
    txt=urllib.request.urlopen(BASE+"/metrics",timeout=30).read().decode(); m={}
    for k in ("prefix_cache_queries","prefix_cache_hits"):
        v=[float(x) for x in re.findall(r'^vllm:%s_total\{[^}]*\} ([0-9.e+]+)$'%k,txt,re.M)]
        m[k]=sum(v) if v else None
    return m
def chat(messages,max_tokens=64):
    body={"model":MODEL,"messages":messages,"max_tokens":max_tokens,"temperature":0,"stream":True,
          "stream_options":{"include_usage":True},"chat_template_kwargs":{"enable_thinking":False}}
    req=urllib.request.Request(BASE+"/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    m0=metrics(); t0=time.time(); first=None; txt=""; pt=0
    with urllib.request.urlopen(req,timeout=1800) as r:
        for line in r:
            line=line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            d=json.loads(line[5:])
            if d.get("choices"):
                c=d["choices"][0]["delta"].get("content") or ""
                if c and first is None: first=time.time()
                txt+=c
            if d.get("usage"): pt=d["usage"]["prompt_tokens"]
    ttft=(first or time.time())-t0; m1=metrics()
    q=m1["prefix_cache_queries"]-m0["prefix_cache_queries"]; h=m1["prefix_cache_hits"]-m0["prefix_cache_hits"]
    return {"prompt_tokens":pt,"ttft_s":round(ttft,2),"hit_tokens":int(h),"hit_rate":round(h/q,3) if q else 0.0,"text":txt}
def show(tag,r): print(f"{tag:<10} prompt {r['prompt_tokens']:>7} tok  TTFT {r['ttft_s']:>6.2f}s  cache hits {r['hit_tokens']:>7} tok ({r['hit_rate']:.1%})  -> {r['text'].strip()[:70]!r}",flush=True)
doc=filler(TOKENS//2)+FACTS+filler(TOKENS//2)
q1=[{"role":"user","content":doc+"\n\nWhat is the relay on bay seven named? Answer in one short sentence."}]
cold=chat(q1); show("cold",cold)
warm=chat(q1); show("warm",warm)
same=cold["text"]==warm["text"]
print(f"REPEAT: TTFT {cold['ttft_s']}s -> {warm['ttft_s']}s ({cold['ttft_s']/max(warm['ttft_s'],1e-6):.1f}x), warm hit rate {warm['hit_rate']:.1%}, "
      f"output {'IDENTICAL' if same else 'DIFFERS'} cold vs warm",flush=True)
conv=[{"role":"user","content":doc+"\n\nWhat is the spare lane token? Answer in one short sentence."}]
ok=True
for turn,(q,want) in enumerate([(None,"9182"),("On which day does the audit close? One short sentence.","Thursday"),
                                ("Repeat the relay name and the spare lane token, nothing else.","Juniper")],1):
    if q: conv+= [{"role":"user","content":"More log: "+filler(TURN_TOKENS)+"\n\n"+q}]  # each turn adds a few blocks, like an agent turn
    r=chat(conv); show(f"turn {turn}",r); conv+=[{"role":"assistant","content":r["text"]}]
    ok&= want.lower() in r["text"].lower() and (turn==1 or r["hit_rate"]>0.5)
print(f"PREFIX_CACHE {'PASS' if (same and warm['hit_rate']>0.5 and ok) else 'FAIL'} (salt {salt})")
