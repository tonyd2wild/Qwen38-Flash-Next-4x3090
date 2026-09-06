#!/usr/bin/env bash
# Tony's simple test: count to 100, single stream, on the 3090 Flash Next endpoint. Waits for the endpoint first.
URL="${URL:-http://100.113.64.18:8090}"; MODEL="${MODEL:-qwen3.8-flash-next}"; LABEL="${1:-3090}"
for i in $(seq 1 360); do curl -s -m 5 -o /dev/null -w "%{http_code}" $URL/v1/models 2>/dev/null | grep -q 200 && break; sleep 5; done
curl -s -m 5 -o /dev/null -w "%{http_code}" $URL/v1/models 2>/dev/null | grep -q 200 || { echo "ENDPOINT NEVER CAME UP $(date +%H:%M:%S)"; exit 1; }
echo "ENDPOINT UP $(date +%H:%M:%S)"; sleep 5
python3 - "$URL" "$MODEL" "$LABEL" <<'PY'
import sys, json, time, urllib.request
url, model, label = sys.argv[1:4]
def run(prompt, max_tokens, tag):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; n = 0; text = ""
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:]); delta = d["choices"][0].get("delta", {}).get("content") or ""
            if delta:
                if first is None: first = time.time()
                text += delta; n += 1
    t1 = time.time()
    # token count from the server's tokenizer-independent chunk count is approximate; count words+punct instead via usage if present
    toks = None
    try:
        body2 = dict(body); body2["stream"] = False
        req2 = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body2).encode(), headers={"Content-Type": "application/json"})
    except Exception: pass
    return {"tag": tag, "ttft_s": round((first or t1) - t0, 3), "wall_s": round(t1 - t0, 3), "chunks": n, "chars": len(text), "text_head": text[:60].replace("\n", " ")}
warm = run("Say OK.", 8, "warmup"); print("warmup", warm)
res = []
for k in range(3):
    # non-stream for exact token counts and decode rate: (completion_tokens) / (wall - ttft_est)
    body = {"model": model, "messages": [{"role": "user", "content": "Count from 1 to 100, separated by commas, nothing else."}], "max_tokens": 400, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); d = json.load(urllib.request.urlopen(req, timeout=600)); wall = time.time() - t0
    ct = d["usage"]["completion_tokens"]; pt = d["usage"]["prompt_tokens"]
    s = run("Count from 1 to 100, separated by commas, nothing else.", 400, "stream%d" % k)
    dec = ct / max(wall - s["ttft_s"], 1e-6)
    res.append(dec); print(f"run {k+1}: completion_tokens={ct} prompt_tokens={pt} wall={wall:.2f}s ttft~{s['ttft_s']}s -> {dec:.1f} tok/s | head: {d['choices'][0]['message']['content'][:40].strip()!r}")
res.sort(); print(f"COUNT100 {label}: median {res[1]:.1f} tok/s (runs {', '.join(f'{x:.1f}' for x in res)})")
PY
