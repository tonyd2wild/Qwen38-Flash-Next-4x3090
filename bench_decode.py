import json, urllib.request
payload = json.dumps({
    "model": "qwen3.8-flash-next-gguf",
    "messages": [{"role": "user", "content": "Write a detailed paragraph about GPU tensor parallelism."}],
    "max_tokens": 200, "temperature": 0,
}).encode()
for run in range(1, 4):
    req = urllib.request.Request("http://localhost:8090/v1/chat/completions",
        data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    t = d["timings"]
    print(json.dumps({"run": run, "prompt_ms": round(t["prompt_ms"], 1),
        "decode_tok_s": round(t["predicted_per_second"], 2),
        "completion_tokens": d["usage"]["completion_tokens"]}))
