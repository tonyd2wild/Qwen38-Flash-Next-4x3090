import json, urllib.request
config = "\n".join([f"  \"service_{i}\": {{\"host\": \"10.0.0.{i}\", \"port\": {8000+i}, \"replicas\": 3, \"timeout_s\": 30, \"retries\": 5, \"log_level\": \"info\"}}," for i in range(1, 25)])
prompt = "Here is a JSON config:\n{\n" + config + "\n}\nOutput the ENTIRE config unchanged except set service_7 port to 9999. Output only the JSON, no explanation. /no_think"
payload = json.dumps({"model": "qwen3.8-flash-next-gguf", "messages": [{"role": "user", "content": prompt}], "max_tokens": 1400, "temperature": 0}).encode()
for run in range(1, 4):
    req = urllib.request.Request("http://localhost:8090/v1/chat/completions", data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    t = d["timings"]
    print(json.dumps({"run": run, "decode_tok_s": round(t["predicted_per_second"], 2), "tokens": d["usage"]["completion_tokens"]}))
