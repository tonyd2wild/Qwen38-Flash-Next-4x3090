import json, urllib.request, sys, time
URL="http://localhost:8090/v1/chat/completions"
PROMPTS = {
 "count100":  ("Count from 1 to 100. Output only the numbers separated by spaces, nothing else. /no_think", 900),
 "code":      ("Write a Python function that implements binary search on a sorted list, with a docstring and type hints. Code only. /no_think", 500),
 "prose":     ("Write a detailed paragraph explaining how tensor parallelism works across multiple GPUs. /no_think", 400),
 "jsonedit":  ("Here is JSON: {\"a\":1,\"b\":2,\"c\":3,\"d\":4,\"e\":5,\"f\":6,\"g\":7,\"h\":8}. Output the ENTIRE JSON unchanged except set \"e\" to 99. JSON only. /no_think", 300),
}
name=sys.argv[1] if len(sys.argv)>1 else "count100"
runs=int(sys.argv[2]) if len(sys.argv)>2 else 3
prompt,maxtok = PROMPTS[name]
payload=json.dumps({"model":"qwen3.8-flash-next-gguf","messages":[{"role":"user","content":prompt}],"max_tokens":maxtok,"temperature":0}).encode()
for r in range(1,runs+1):
    req=urllib.request.Request(URL,data=payload,headers={"Content-Type":"application/json"})
    t0=time.perf_counter()
    with urllib.request.urlopen(req,timeout=600) as resp: d=json.load(resp)
    t=d.get("timings",{})
    print(json.dumps({"task":name,"run":r,
      "decode_tok_s":round(t.get("predicted_per_second",0),2),
      "prompt_ms":round(t.get("prompt_ms",0),1),
      "tokens":d["usage"]["completion_tokens"],
      "wall_s":round(time.perf_counter()-t0,2)}))
