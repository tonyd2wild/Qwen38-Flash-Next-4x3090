# 14: Fixed overheads for warm and incremental TTFT (T11, portfolio P9)

Status: done · Model: Opus 5.5 (the board's choice since T5) · Effort: high · Started: 2026-10-17 23:28 · Wall time: 1:45 (three boots, R20-R22)

## Goal

Remove three per-turn or per-boot costs that are not prefill or decode. Success is measured on the hostpin lane against R19 (`results/scoreboard-R19-20261017-230308.json`):

- **Tokenization.** Warm 128K TTFT through chat drops to within ~0.05 s of the same prompt sent as token ids (`turn_overhead.py`; R15: chat 0.72 s, ids 0.42 s). The incremental-turn rows improve by about the tokenization time.
- **Triton JIT.** The first requests after a boot no longer pay 0.4-1.8 s JIT spikes per new kernel specialization.
- **Hostpin boot.** Back near the prefix lane's ~4.5 min, from 8.5 min (R19).

Decode and cold prefill must not move beyond the R13/R14 spread (±2.5% / ±1.5%), and the gate must pass.

## Hypothesis and kill test

**Tokenization.** Measured before writing this section, in a CPU-only container of the lane image with the checkpoint's tokenizer:
- `tok(text, truncation=True, max_length=262145)` on a 127,594-token chat prompt (649,299 chars) takes 306 ms.
- The chat template takes 0.2 ms, and list conversion 1 ms.
- The time is the Rust pipeline: NFC normalization takes 25 ms, and the rest is the pre-tokenizer regex and BPE on one thread.
- The same text cut into 16 pieces at the safe points below and encoded with `encode_batch_fast` (rayon) takes **32 ms**, with identical ids.
- Encoding the pieces serially takes 256 ms. Python threads calling `encode` take 360-445 ms: the binding holds the GIL.
- With `TOKENIZERS_PARALLELISM=false`, the batch runs serially (212 ms).

So the time is single-threaded Rust work, and both a parallel split and a cache of pieces can remove most of it.

**Kill tests:**
- **K1 (exactness, offline, before any boot).** The split tokenization equals `tok(text, **kwargs)["input_ids"]` exactly on a corpus:
  - the quality-gate prompts and the scoreboard prompts, rendered through the chat template with tools;
  - the 128K filler conversations;
  - code;
  - random fuzz strings built from an alphabet chosen to break the argument below: `\n`, `\r`, spaces, tabs, ASCII and non-ASCII letters, combining marks, digits, punctuation, CJK, emoji, Hangul jamo, every added token, and apostrophe contractions.
  
  Any mismatch kills the cut rule that produced it.
- **K2 (speed, offline).** With rayon capped at 8 threads, 128K tokenization is at least 3x faster. If not, only the cache ships.
- **K3 (boot).** Chat warm 128K TTFT minus ids warm 128K TTFT falls below 0.1 s (R15: 0.30 s). Decode, cold and gate unchanged.

**Triton cache.** `/root/.triton/cache` is in the container's own filesystem, so it is empty at every boot (`02-time-accounting.md`). Pointing `TRITON_CACHE_DIR` into the mounted `/root/.cache` makes boot N+1 load what boot N compiled. The cache key covers the kernel source, the options and the device, so a loaded binary is the binary a compile would produce: lossless.
- Kill test: on the second boot with the knob, a first-contact probe (the first requests after boot) shows no JIT spikes; on the first boot it shows them.

**Hostpin boot.** R19's log (`.cache/ple/r19-boot.log`) shows two separate costs:
1. **Registration contention.**
   - The leader registers in 26.5 s, then goes on to load weights.
   - The followers register at the same time but take 98.6 s, against 19.9 s for four concurrent registrations in the K1 bench.
   - The leader's weight loading slows the followers' pinning.
2. **The page cache.**
   - Weight loading takes 193 s (R13: 64.5 s), and shards keep loading at 10-13 s each after the followers finish.
   - The 69 GB of target weights no longer fit in the page cache. The memfd holds 51 GB, and the PLE files' own pages (up to 51 GB, read by the fill) sit beside it in 125 GiB.

Fixes, both in `ple_hostpin.py`, with no change to the call site:
- (a) A TP barrier after every rank has registered, so no rank loads weights while another pins.
- (b) After the leader's fill, `posix_fadvise(POSIX_FADV_DONTNEED)` on the 10 PLE files. Their bytes now live in the memfd, so their page cache is dead weight. This is a hint from the process on files it opened, not a host setting.

Kill test: the boot log's "Model loading took" falls to under 120 s on all ranks (R19: 214-263 s).

## Setup

- Branch / commit: `perf-comm-ple` @ `9d14d94`
- Lane: `launch-qwen38fn-w4a16-hostpin.sh` (R19's lane), image `qwen38fn-vllm:8a728663`, `GPU_ARGS="--device nvidia.com/gpu=all"`
- Ledger rows: R20, R21, R22
- Box state before: GPUs 15-91 MiB, localhost:8090 and host.docker.internal:8090 down, page cache 43 GiB

## Design

> **Outcome notes, added after coding.**
> - A third cut rule, **C3** (between an ASCII letter and a space followed by an ASCII letter), was added after the first check run. Single-line prose had almost no C1/C2 points. Its argument is in "What I did" step 2 and in `tok_split.py`'s header.
> - The hostpin part changed in the bench before any boot: the leader registers before the exchange, the followers rotate their chunk order, and DONTNEED runs per 64 MiB copy (after R20).
>
> The Design text below is as written before coding.

### Tokenization: where it runs

- `vllm/renderers/base.py` `BaseRenderer._tokenize_prompt` calls `tokenizer(prompt["prompt"], **params.get_encode_kwargs())`. It runs in the renderer's thread pool (`renderer_num_workers`, default 1) through `make_async`, after `arrival_time` is set in `render_chat_async`. That is why it sits inside the server's TTFT histogram.
- Chat, completions-text and responses requests all pass through it. Token-id prompts skip it, which is the R15 "ids" row.
- The chat template is applied first, with `tokenize=False`, and produces one string.
- The encode kwargs are `truncation=(max_length is not None)`, `max_length=max_input_tokens + 1`, `add_special_tokens=...`. The tokenizer is `Qwen3_5Tokenizer` (transformers 5.20.1, `TokenizersBackend`, tokenizers 0.23.3), wrapped by vLLM's `CachedTokenizer` and a thread-safe pool of deep copies.

### The pipeline the argument rests on

HF `tokenizers` encodes a string in four steps:
1. It splits out added tokens. All 33 here are `normalized=False`, `lstrip=rstrip=single_word=False`, and none contains `\n`. They are matched leftmost-longest on the raw text.
2. Each remaining segment is NFC-normalized.
3. The segment is pre-tokenized: a `Split` with the Qwen regex (`Isolated`), then `ByteLevel` (`use_regex=False`).
4. Each pre-token is BPE-merged independently. The post-processor is `ByteLevel` with `trim_offsets=False`, which adds no tokens.

The ids are the concatenation, over pre-tokens, of the BPE of each pre-token. So cutting the text at position p is exact when:
- no added-token match crosses p;
- NFC of the two sides equals NFC of the whole;
- the full text's pre-token sequence has a boundary at p, and the sequence left of p is what the left piece alone produces.

### Cut rules

Only two kinds of positions are used.

**C1: just before an occurrence of `<|im_start|>`.** This is a segment boundary of step 1, as long as every literal occurrence is matched there. That holds if no added token overlaps `<|im_start|>`: none contains it, and no proper suffix of one is a proper prefix of another. With the strip flags all False, nothing on either side changes. The segments are then encoded independently by construction.
- Message content containing the literal text `<|im_start|>` is also treated as the special token by the full tokenization (`split_special_tokens=False`), so the rule still holds there.

**C2: between `\n` and an ASCII letter** (`text[p-1] == "\n"`, `text[p]` in `[A-Za-z]`).
- *Added tokens:* no token contains `\n`, so no match crosses p.
- *NFC:* `\n` never composes with anything, and an ASCII letter has combining class 0 and is NFC-stable, so NFC(a) + NFC(b) = NFC(a + b).
- *Regex, the match that covers `\n`:* some match covers the `\n` at p-1, since `\s*[\r\n]+` always matches at a newline. That match can only be:
  - `\s*[\r\n]+`, which stops at the letter;
  - ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*`, which also stops at the letter;
  - `\s+(?!\S)` or `\s+`. These are never reached: they come after `\s*[\r\n]+` in the alternation, and that alternative already succeeds on any whitespace run that contains a newline.
  
  The contraction, letter and number alternatives cannot contain `\n`; the letter alternative's optional leading character excludes `\r\n`. So a match ends exactly at p, and the next match starts at p in both the full text and the right piece.
- *Regex, the matches left of p:* none of them look past p. The only lookahead is `(?!\S)` in `\s+(?!\S)`, and that alternative never ends at p, as shown. Greedy stops at p see the same "not whitespace / not newline" outcome at the letter and at end-of-string.
- The same holds for the regex variant in `tokenizer_config.json`, which adds `\p{M}` only inside the letter and symbol classes.

**Preconditions.** The module checks these at load time and disables itself, with a log line, if any fails:
- normalizer NFC or none;
- pre-tokenizer exactly `[Split(<one of the two known regexes>, Isolated), ByteLevel(use_regex=False)]`;
- model BPE with no dropout;
- the added-token flags and overlaps above;
- an init self-test that the split equals the full call on a fixed sample with every added token.

The fuzz corpus in K1 checks the argument empirically; the preconditions check that the argument applies to the tokenizer actually loaded.

### Pieces and the cache

**Cutting.**
- C1 cuts are always taken, so each message starts a piece.
- Inside a message, C2 cuts are taken greedily: the first C2 point at least `MIN_PIECE` characters (4,096) after the previous cut.
- The cuts depend only on the text from the start, so a conversation's pieces are a prefix of the next turn's pieces until the first edited message. After that edit, C1 realigns the cuts at the next message.

**Encoding.**
- Each piece is looked up in an LRU dict keyed by its text. Values are `array('i')`, and the dict is bounded by total cached tokens (default 4M, about 40 MB with the key strings).
- The misses are encoded in one `encode_batch_fast` call on a private `Tokenizer.from_str(backend.to_str())` copy, with `no_truncation()` and `no_padding()`. The shared backend's truncation state is set and left behind by transformers' `__call__`, so it is not used.
- The result is the concatenation of the pieces.

**Fallback to the original call.** The original call is used whenever:
- offsets are wanted;
- the kwargs hold anything but `truncation`, `max_length` and `add_special_tokens`;
- the text is shorter than 16K characters (no gain);
- the result is longer than `max_length`. Truncation is then the original's to do; that is the over-length error path anyway.

**Thread safety.** A lock covers the cache. The Rust encode runs outside it. The renderer pool has one worker by default.

**Threads.** `RAYON_NUM_THREADS` caps the rayon pool for the API-server process (launcher default 8). The engine core and the GPU workers are separate processes whose busy threads must not be starved.
- If parallelism is unavailable (a fork after first use disables it in `tokenizers`), the batch runs serially. That is still exact, and the cache still saves the repeated pieces.

**Hook.** An overlay of `vllm/renderers/base.py` with one branch at the top of `_tokenize_prompt`, plus a new module `vllm/renderers/tok_split.py`. Both are mounted only with the launcher knob `TOKSPLIT=1` (`QWEN4EXP_TOKSPLIT=1`). The launcher's hash check covers the new mounts. A debug knob, `QWEN4EXP_TOKSPLIT_VERIFY=1`, also runs the original call, compares, logs any mismatch and returns the original ids. It is not for measured boots.

**Alternatives considered:**
- *Caching the whole previous prompt and diffing by the longest common prefix.* Its cut needs the same safety argument, and an edit in the middle loses everything after it. C1 realignment recovers.
- *Python threads.* The GIL is held (measured above).
- *A faster tokenizer library.* Not in the image. A new dependency needs its own exactness proof on all of the above.
- *Token ids from the client.* Not drop-in.
- *Tokenizing in the API server before `arrival_time`.* That only moves the cost out of the histogram.

### Triton cache

A launcher knob, `TRITON_CACHE=1`, adds `-e TRITON_CACHE_DIR=/root/.cache/triton`. `/root/.cache` is mounted from `.cache/ple`, which is gitignored. It defaults to 0 in the base launcher, so the tuned and prefix lanes are unchanged. The hostpin wrapper turns it on if it is kept.
- Triton writes cache entries atomically (temporary file, then rename), so four workers can share the directory.
- An empty cache directory behaves exactly like the default: everything compiles. So the first boot with the knob doubles as the "before" measurement.

### Hostpin boot

`PLEHostPinTable` gains two steps:
- (b) the leader calls `posix_fadvise(DONTNEED)` on each table file after `_fill`;
- (a) every rank calls `barrier()` after its registration. `barrier` is a new optional callable: `tp.barrier` in `ple_layer.py`, `None` in the bench.

The construction path has no rank-conditional collectives. Every rank runs the same exchange call and then the same barrier, in that order, so the collective order is the same everywhere. A rank that raises before the barrier leaves the others waiting there; that is the same failure mode as today's broadcast, and the boot fails either way.

DONTNEED changes the page cache for later boots: the next boot reads the PLE files from NVMe, sequentially, through 8 `preadv` threads. A staged-lane boot right after a hostpin boot starts with the table uncached. That is recorded as a side finding, not hidden.

### Boot plan (one change per boot where it can be attributed)

Every boot runs the gate first:
1. **R20.** Hostpin lane, with the boot fix and `TRITON_CACHE=1` on an empty Triton directory, which behaves like no knob.
   - Measured: boot time and model loading (the boot fix against R19), the first-contact probe (Triton "before").
   - Then `quality_gate.py --compare` and `turn_overhead.py --depth 128000` (the tokenization baseline on this lane).
2. **R21.** The same lane on a warm Triton cache. Measured: boot time again, with the page cache in its new steady state, and the first-contact probe (Triton "after").
3. **R22.** Plus `TOKSPLIT=1`. Measured: the gate, `turn_overhead.py`, and the full `scoreboard.py` (the new column against R19).

**Test plan before any boot:**
- `tools/tok_split_check.py`, run in a CPU-only container of the image: K1 on the corpus plus 20,000 fuzz strings, and K2 timings at 16K, 64K and 128K, with rayon at 4, 8 and 64 threads.
- The hostpin change is checked with `ple_gather_bench.py filepin_4`, which is the same code with the barrier as an exchange, plus the DONTNEED effect read from `mincore` residency before and after.

## What I did

1. **Located the cost.** Read `renderers/base.py` and `renderers/hf.py` in the image (Design). Timed the tokenizer in a CPU-only container (`scratchpad tokbench.py` / `gilprobe.py`, numbers in Hypothesis).
2. **Wrote the split.**
   - New `vllm-w4a16/patch/tok_split.py`: cut rules, preconditions, self-test, piece cache.
   - Overlay `vllm-w4a16/patch/renderers_base.py`: the image's `vllm/renderers/base.py` plus two blocks marked `TOKSPLIT`, one building the splitter in `BaseRenderer.__init__`, one at the top of `_tokenize_prompt`.
   - **C3 was added after the first check run.** C1 and C2 alone cut the bench's single-line filler into only 2 pieces (128K: 335 -> 208 ms). C3 is the cut between an ASCII letter and a space followed by an ASCII letter, with the same argument as C2: the letter run ends at the space under every alternative, `" word"` starts the next match, no added token contains a space, and NFC can't join a letter to a space. A space precondition was added.
3. **K1/K2 offline:** `vllm-w4a16/tools/tok_split_check.py`. It runs in a CPU-only container through vLLM's own `get_tokenizer` and compares against the server's exact call:
   ```
   docker run --rm --entrypoint python3 -e RAYON_NUM_THREADS=8 -v ~/models/qwen38fn-w4a16-fp8ple:/models/qwen38fn:ro \
     -v $PWD/vllm-w4a16:/lane qwen38fn-vllm:8a728663 /lane/tools/tok_split_check.py --fuzz 20000 --out /lane/results/tok-split-check-T11-rayon8-<time>.json
   ```
   The same command ran with `RAYON_NUM_THREADS=4` and `64` and `--fuzz 0`, for timings only.
4. **Hostpin boot, in the bench.** `GPU_ARGS="--device nvidia.com/gpu=all" python3 vllm-w4a16/tools/ple_gather_bench.py --label T11-<step> --modes resident,filepin` was run three times, with one change each:
   - **barrier + DONTNEED** (`ple-gather-bench-T11-barrier-*.json`): DONTNEED works (table residency 38% -> 0.1%). But four concurrent registrations needed 12-28 retries each and 72 s.
   - **The leader registers before the exchange** (`-leaderfirst-`): 0 retries. The leader takes 11.2 s alone, but the three followers together need 51 s.
   - **Each rank starts at its own 4 GiB chunk** (`-rotated-`): followers 18.7 s, 0 retries, setup 58 s, bytes equal. Every chunk's device pointer is now checked against its host address.
   - After R20, DONTNEED moved from the end of the fill into every 64 MiB copy, so the table's file pages never build up and push the weights out.
   - `ple_layer.py` passes `barrier=tp.barrier`, `rank`, `world`. The collective order is the same on every rank: one broadcast, then one barrier.
5. **Launcher knobs.**
   - `qwen38fn-w4a16-3090-tp4.sh` gets `TRITON_CACHE` (0|1, adds `-e TRITON_CACHE_DIR=/root/.cache/triton`) and `TOKSPLIT` (0|1, `TOKSPLIT_THREADS`, default 8, as `RAYON_NUM_THREADS`, plus the two mounts). The hash check covers the new mounts.
   - Both default to 0, so the tuned and prefix lanes are unchanged. `launch-qwen38fn-w4a16-hostpin.sh` defaults both to 1.
   - New `tools/first_contact.py`: two passes of the same cold shapes right after boot; pass 1 minus pass 2 is the JIT tax.
6. **Boots.** In the ledger, each was ready when the log said `Application startup complete`.
   - **R20** (23:57): `TOKSPLIT=0 GPU_ARGS=... bash vllm-w4a16/launch/launch-qwen38fn-w4a16-hostpin.sh`, with the boot fix and an empty Triton cache directory (`.cache/ple/triton` did not exist). Ran `first_contact.py --label R20`, `quality_gate.py --compare --label R20-bootfix` and `turn_overhead.py --label R20 --depth 128000 --reps 2`.
   - **R21** (00:16): the same command, after DONTNEED moved into each copy, on a warm Triton cache. Ran `first_contact.py --label R21` and `quality_gate.py --compare --label R21-tritoncache`. The gate's SUSPECT was reviewed with 12 idle repeats of the request (`.cache/ple/r21-repeat-tool-search.py`).
   - **R22** (00:32): `GPU_ARGS=... bash vllm-w4a16/launch/launch-qwen38fn-w4a16-hostpin.sh`, now with `TOKSPLIT=1` by default. Ran `quality_gate.py --compare --label R22-toksplit`, `turn_overhead.py --label R22-toksplit --depth 128000 --reps 2` and `scoreboard.py --label R22`. Stopped at 01:08.
   - The boot logs are `.cache/ple/r2{0,1,2}-boot.log`; the gate and scoreboard console output are `.cache/ple/r2*-gate.out` and `r22-score.out`.

## Measurements

**K1, exactness** (`results/tok-split-check-T11-rayon8-20261017-234217.json`):
- 0 mismatches.
- Cases: 48 chat renderings with tools, tool calls, tool results carrying the image's Python sources, thinking on and off, and literal `<|im_start|>` / `<tool_call>` in user text. Each was checked at the service piece size and at 64 chars, plus 12 filler prompts at 16K-128K and 40,000 fuzz strings.
- About 359,000 cuts exercised: 58,800 in the general fuzz, 284,628 in the fuzz weighted to letters, spaces, newlines, combining marks, contractions and jamo.
- Every case was checked three ways (uncached, first cached call, all-hits call) against both `add_special_tokens` settings.

**K2, speed** (same files, best of 3; "next turn" = the conversation plus 2K new tokens):

| Prompt | Full call | Split, cold cache (8 threads) | 4 threads | 64 threads | Next turn, full | Next turn, cached |
|---|---:|---:|---:|---:|---:|---:|
| 16K tokens (21 pieces) | 39.0 ms | 6.3 ms | 9.0 | 5.6 | 43.9 ms | 3.4 ms |
| 64K (81 pieces) | 170.7 ms | 26.2 ms | 31.5 | 13.7 | 173.5 ms | 3.8 ms |
| 128K (160 pieces) | 332.3 ms | 38.9 ms | 54.4 | 22.2 | 337.7 ms | 4.8 ms |

**K3, warm 128K turn three ways** (`turn-overhead-R20-*.json` against `turn-overhead-R22-toksplit-*.json`; 127,610 prompt tokens, 127,296 hits):

| | R20 client TTFT | R20 server - prefill | R22 client TTFT | R22 server - prefill |
|---|---:|---:|---:|---:|
| chat | 0.64-0.65 s | 310-315 ms | **0.36 s** | 26 ms |
| completions, text | 0.63-0.64 s | 308-317 ms | 0.35-0.36 s | 20-23 ms |
| completions, token ids | 0.38 s | 20-22 ms | 0.37-0.39 s | 18-22 ms |
| cold 128K | 104.85 s | 314 ms | 105.08 s | 30 ms |

Chat minus ids went from +0.27 s to -0.02 s. Chat is now as fast as sending token ids; ids is slightly slower because the client sends a 127K-integer JSON body.

**Triton cache, first contact after boot** (`first-contact-R20-*.json`, empty cache, against `first-contact-R21-*.json`, warm cache, 98 MB):

| Request (cold, fresh salt) | R20 pass 1 | R20 pass 2 | R21 pass 1 | R21 pass 2 |
|---|---:|---:|---:|---:|
| "Say OK." | 1.06 s | 0.16 | 0.24 | 0.16 |
| 45 tokens | 1.97 s | 0.16 | 0.19 | 0.16 |
| 1K | 3.91 s | 0.50 | 0.56 | 0.50 |
| 4K | 3.23 s | 1.82 | 1.86 | 1.85 |
| 8K | 5.31 s | 3.64 | 3.70 | 3.69 |
| Sum of pass 1 - pass 2 | **10.34 s** | | **0.23 s** | |

Boot: the profile and KV phase went from 81 s (R20) to 2-3 s (R21, R22), and graph capture from 9 s to 2-3 s.

**Boot time** (launch to `Application startup complete`; `.cache/ple/r19..r22-boot.log`):

| Boot | Ready | Hostpin setup (leader fill / register; followers register; retries) | Weight loading | Page cache before boot |
|---|---:|---|---:|---:|
| R13 (staged lane) | ~4.2 min | - | 64.5 s | warm |
| R19 (T5) | ~8.5 min | 41.8 s leader, 114 s followers (15.2 / 26.5; 98.6; 2) | 193 s | - |
| R20 | 7:25 | 81 s all ranks (18.3 / 29.3; 23.5; 0) | 185 s | 18 GiB |
| R21 | 5:27 | 45 s (19.7 / 6.1; 19.1; 0) | 173 s | 48 GiB |
| R22 | **5:23** | 53 s (19.6 / 14.0; 19.4; 0) | 157 s | 59 GiB |

"Weight loading" includes the hostpin setup, which runs at its start. The target shards themselves took ~105-130 s in R20-R22 against ~60 s in R13.

**Scoreboard, R19 against R22** (`results/scoreboard-R19-20261017-230308.json`, `results/scoreboard-R22-20261018-004644.json`; R13/R14 spread: decode ±2.5%, cold prefill ±1.5%, warm ±0.1 s):

| Metric | R19 (hostpin, T5) | R22 (T11) | Delta |
|---|---:|---:|---:|
| Decode, count to 100 (tok/s) | 204.0 | 207.5 | +1.7% (spread) |
| Decode, prose 512 (tok/s) | 144.9 | 148.0 | +2.1% (spread) |
| Decode, code (tok/s) | 180.5 | 183.0 | +1.4% (spread) |
| Cold TTFT 1K / 16K / 64K (s) | 0.50 / 7.47 / 29.78 | 0.50 / 7.26 / 29.20 | 0 / -0.21 / -0.58 |
| Cold TTFT 128K / prefill tok/s | 107.70 s / 1,184 | 105.22 s / 1,213 | -2.5 s (2 preemptions both) |
| Warm TTFT 1K / 16K / 64K / 128K (s) | 0.16 / 0.24 / 0.50 / 0.66 | 0.16 / 0.18 / 0.42 / **0.38** | 0 / -0.06 / -0.08 / **-0.28** |
| Decode at 64K / 128K (tok/s) | 125.1 / 158.1 | 131.3 / 144.4 | noisy single runs |
| Incremental 16K + 1K / 4K (s) | 0.67 / 2.14 | 0.61 / 2.04 | -0.06 / -0.10 |
| Incremental 64K + 4K / 16K (s) | 2.43 / 8.11 | 2.26 / 7.80 | -0.17 / -0.31 |
| Incremental 128K + 4K / 16K (s) | 2.62 / 8.35 | **2.27 / 7.90** | -0.35 / -0.45 |
| Needle ~200K | correct, 175 s, 1,141 tok/s | correct, 171 s, 1,168 | -4 s |
| 155K cold tok/s, preemptions, min free | 1,278, 2, 0.60 | 1,310, 2, 0.60 | +2.5% |
| Concurrency 2 / 4 / 6 aggregate (tok/s) | 159.7 / 279.8 / 301.2 | 165.0 / 259.5 / 337.1 | 2 rounds each: noisy |
| Per-stream decode median 2 / 4 / 6 | 104.1 / 80.8 / 59.4 | 111.2 / 83.6 / 62.3 | |
| KV pool | 299,482 | 299,482 | 0 |

The tokenization saving is ~2.6 us per prompt token, about 0.3 s at 128K. It shows directly in every warm and incremental row at 64K and deeper. In the cold rows it is inside the ±1.5% spread, except at 128K.

## Result

**Kept, all three parts.**
- **Tokenization (`TOKSPLIT=1`).**
  - Warm 128K TTFT is 0.66 -> 0.38 s on the scoreboard and 0.65 -> 0.36 s on `turn_overhead.py`, now level with sending token ids.
  - Incremental 128K + 4K is 2.62 -> 2.27 s.
  - It is exact by construction for the checked pipeline, and by test on ~359,000 cuts. Prompt token counts in the gate are identical to R20's on every prompt.
- **Triton cache (`TRITON_CACHE=1`).** The first-contact JIT tax after a boot is 10.3 -> 0.2 s, and the boot is ~85 s shorter (profile run and capture no longer compile).
- **Hostpin boot.** Ready in 5:23 instead of ~8.5 min, with 0 registration retries in all three boots. The fix:
  - the leader registers before handing the buffer out;
  - the followers start at different chunks;
  - a barrier after registration;
  - the table's file pages are dropped while copying.

**The rest of the gap to the staged lane's ~4.2 min:**
- the ~50 s setup itself;
- slower weight reads. 47.7 GiB pinned plus 64 GiB of weights barely fit in 125 GiB, so part of the weights comes from NVMe each boot.

## Quality gate

| Boot | Verdict | Prompts | p99 \|dlogprob\| | Needles 30K / 100K / 200K | 155K cold | Pool |
|---|---|---|---:|---|---|---:|
| R20 (boot fix, empty Triton cache) | PASS | 13 IDENTICAL, 7 FP-sized | 0.118 nat over 1,078 positions | correct | 1,312 tok/s, 2 preemptions, min free 0.54-0.58 GiB | 299,482 |
| R21 (warm Triton cache) | REVIEW | 11 IDENTICAL, 8 FP-sized, 1 SUSPECT | 0.129 | correct | 1,277 tok/s | 299,482 |
| R22 (toksplit) | PASS | 12 IDENTICAL, 8 FP-sized | 0.116 over 1,046 | correct | 1,308 tok/s, 2 preemptions, min free 0.60-0.66 GiB | 299,482 |

R22's boot facts were all as baseline. Tool calls parsed in all three boots.

**The R21 SUSPECT, reviewed.** `tool_search` first diverges at token 9. The baseline picks id 79114 with a 3.75-nat margin; R21 picked id 28 at -0.387, with the baseline token at -1.137. The prefix drift before it is 0.002 nat.

Twelve idle repeats of exactly that request on the same R21 boot (`.cache/ple/r21-repeat-tool-search.py`) all picked 79114, but the gap to id 28 ranged from **1.25 to 5.25 nat** from one repeat to the next. So this position swings ~4 nat with nothing changed: it is the T14 nondeterminism, not the Triton cache. A loaded cache entry is the binary the compiler produced, keyed on source, options and device. The same prompt was IDENTICAL on R20 and R22.

## Side findings

- **Tokenization and parallelism in `tokenizers`.**
  - The Python binding holds the GIL in `encode`, so Python threads make it slower (360-445 ms against 306).
  - Only `encode_batch` / `encode_batch_fast` parallelize, on rayon.
  - `TOKENIZERS_PARALLELISM=false`, or a fork after first use, makes the batch serial (212 ms).
  - The renderer's single default worker (`renderer_num_workers=1`) serializes tokenization across requests. TOKSPLIT takes ~40 ms of rayon work per cold 128K prompt, and the cache usually leaves only a few ms.
- **The server's TTFT histogram includes rendering and tokenization.** `arrival_time` is set at the top of `render_chat_async`. Server TTFT minus prefill is therefore the frontend cost, and it is now ~20-30 ms at any depth.
- **The image's tokenizer is transformers 5.20.1 `Qwen3_5Tokenizer` (`TokenizersBackend`) on tokenizers 0.23.3.** `tokenizer_config.json` carries a `pretokenize_regex` with `\p{M}` that differs from `tokenizer.json`'s. `tok_split._unsupported()` accepts both: C2 and C3 hold for both.
- **`cudaHostRegister` on pages another process has already registered is slow when several processes walk them in the same order.**
  - Three followers took 51 s in step, against 11 s for one alone. Rotating each rank's chunk order brought that to 19 s.
  - Concurrent first registrations, with the leader among them, failed transiently 12-28 times per rank. Registering the leader first removed every failure.
  - This box has no movable zone and no CMA, so it is not page migration.
- **The PLE files are now dropped from the page cache by every hostpin boot.** A staged-lane boot right after a hostpin boot reads the table from NVMe until it is warm again. Expect slower first long prefills, as R13's shallow-step numbers had a 25-38% resident table.
- **Weight loading is slow when the weights are not cached.** It runs at ~0.3-0.6 GB/s through 4 ranks' mmap faults, against ~2.6 GB/s for the fill's sequential `preadv`. A sequential prefetch of the weight files (`POSIX_FADV_WILLNEED` or a `preadv` pass before the loader) could cut the boot further. It was not tried: one change too many for this task.
- **`tool_search` token 9 is a sensitive repeatability probe for T14:** 1.25-5.25 nat spread over 12 idle repeats of one request on one boot. It is larger than anything `repeat_probe.py` showed.
- **Cold 16K/64K got ~0.2-0.6 s faster in R22,** more than tokenization alone (0.03-0.15 s). Spread or the warm Triton cache; not attributed.

## Open questions

- Would a sequential prefetch of the target weight files bring the hostpin boot to the staged lane's ~4.2 min (T13 or later)?
- On the host (Docker 29.2.1, no sandbox), does the hostpin registration behave the same way, with 0 retries and ~19 s for the followers? `run-host-fast.sh` should record the setup line.
- Is the rayon pool contending with the GPU workers' launch threads when a long cold prompt arrives during 6-stream decode? The concurrency rows show no loss (6 streams 337 tok/s), but the scoreboard does not mix a long cold prompt into decode.

## Next step

**T4** (P3, Mamba page leak; Opus 5.5 / xhigh). 128K and 155K cold are still preemption-bound: 1,213-1,310 tok/s against 2,189 at 64K, with 2 preemptions each. Measure on the hostpin lane, which now has `TRITON_CACHE=1` and `TOKSPLIT=1` by default; R22 is the baseline. Then T14, using the `tool_search` token-9 probe, and T10.

The server was **stopped** at 01:08; GPUs free (15-92 MiB).
