$ ./bench-host.py --sizes 1000,16000,64000,128000 --runs 5 --gen 512
waiting for http://localhost:8090 ...
bench 2026-10-17 17:55:38  http://localhost:8090  model qwen3.8-flash-next

== Decode, short prompt, 5 runs, max 512 tokens
  run 1: 391 tokens, TTFT 175 ms, decode 82.5 tok/s
  run 2: 391 tokens, TTFT 232 ms, decode 143.6 tok/s
  run 3: 391 tokens, TTFT 215 ms, decode 142.5 tok/s
  run 4: 391 tokens, TTFT 221 ms, decode 143.9 tok/s
  run 5: 391 tokens, TTFT 223 ms, decode 143.7 tok/s
  DECODE median 143.6 tok/s, TTFT median 221 ms
  prose: 512 tokens, TTFT 219 ms, decode 110.4 tok/s

== TTFT by prompt size (cold = fresh prefix, warm = same prompt again)
    target   prompt   cold TTFT      prefill   warm TTFT  warm hits  decode@ctx
      1000    1,038       2.63s      394 t/s       1.87s     48,256     7.4 t/s
     16000   15,964      18.89s      845 t/s       0.25s     15,808     2.9 t/s
     64000   63,831      45.63s    1,399 t/s       0.67s     63,232   125.0 t/s
    128000  127,768     152.62s      837 t/s       0.85s    127,296   136.6 t/s

  server preemptions so far: 2
log: /home/xgenadam/workspace/AI/Qwen38-Flash-Next-4x3090/bench-20261017-174748.log
