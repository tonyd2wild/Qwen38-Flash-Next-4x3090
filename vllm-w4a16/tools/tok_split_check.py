#!/usr/bin/env python3
# Offline check of patch/tok_split.py (T11, journal/14-p9-fixed-overheads.md), no GPU and no server. Runs inside the
# lane image with the checkpoint's tokenizer, through vLLM's own get_tokenizer, and compares TokSplit against the call
# the server makes (tok(text, truncation=True, max_length=262145, add_special_tokens=...)):
#   exact   agent-style chats rendered by the chat template (system + tools + tool calls + tool results carrying the
#           image's own Python sources), the 16K-128K filler prompts, and --fuzz random strings from an alphabet
#           built to break the cut argument; pieces cut at --fuzz-piece chars so every string has many cuts
#   speed   full call against split (cold piece cache) and split on the next turn (cache warm, 2K new tokens), at
#           16K / 64K / 128K; rayon threads from RAYON_NUM_THREADS (set it per run)
# Exit code 1 on any mismatch.
# Usage (repo root):
#   docker run --rm --entrypoint python3 -e RAYON_NUM_THREADS=8 -v ~/models/qwen38fn-w4a16-fp8ple:/models/qwen38fn:ro \
#     -v $PWD/vllm-w4a16:/lane:ro qwen38fn-vllm:8a728663 /lane/tools/tok_split_check.py --fuzz 20000 [--out /lane/...]
import argparse, glob, importlib.util, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchlib as B

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/models/qwen38fn")
ap.add_argument("--patch", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "patch", "tok_split.py"))
ap.add_argument("--fuzz", type=int, default=20000)
ap.add_argument("--fuzz-piece", type=int, default=8)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--skip-speed", action="store_true")
ap.add_argument("--out", default="")
a = ap.parse_args()

from vllm.tokenizers import get_tokenizer
spec = importlib.util.spec_from_file_location("tok_split", a.patch)
TS = importlib.util.module_from_spec(spec); spec.loader.exec_module(TS)

tok = get_tokenizer(a.model, trust_remote_code=True)
MAXLEN = 262145
def ref(text, ast=False):
    return list(tok(text, truncation=True, max_length=MAXLEN, add_special_tokens=ast)["input_ids"])

t0 = time.perf_counter()
split = TS.make_tok_split(tok)
assert split is not None, "TOKSPLIT disabled (see the log line above)"
R = {"rayon_threads": os.environ.get("RAYON_NUM_THREADS", "unset"), "init_s": time.perf_counter() - t0, "exact": {}, "speed": []}
print(f"init {R['init_s']:.2f} s, rayon threads {R['rayon_threads']}")
rng = random.Random(a.seed)
fails = []

def check(kind, text, piece=None):
    mp = split.min_piece
    if piece is not None: split.min_piece = piece
    try:
        got = split.encode(text, use_cache=False)
        got2 = split.encode(text)  # through the cache, twice: the second call is all hits
        got3 = split.encode(text)
    finally:
        split.min_piece = mp
    for ast in (False, True):
        want = ref(text, ast)
        for g in (got, got2, got3):
            if g != want:
                n = next((i for i, (x, y) in enumerate(zip(g, want)) if x != y), min(len(g), len(want)))
                fails.append({"kind": kind, "ast": ast, "at": n, "len": len(want), "text": text[:2000]})
                print(f"MISMATCH {kind} add_special_tokens={ast} at token {n} of {len(want)}: {text[max(0, n * 3 - 80):n * 3 + 80]!r}")
                return False
    R["exact"].setdefault(kind, 0); R["exact"][kind] += 1
    R.setdefault("cuts", {}).setdefault(kind, 0)
    if piece is not None: split.min_piece = piece
    R["cuts"][kind] += len(split.pieces(text)) - 1
    split.min_piece = mp
    return True

# 1. Agent-style chats, rendered by the server's chat template.
srcs = sorted(glob.glob("/usr/local/lib/python3.12/dist-packages/vllm/renderers/*.py"))[:12] or [__file__]
files = [(os.path.basename(p), open(p).read()) for p in srcs]
for c in range(12):
    msgs = [{"role": "system", "content": B.SYSTEM}, {"role": "user", "content": B.TASKS[c % len(B.TASKS)]}]
    for t in range(1 + c):
        name, body = files[(c + t) % len(files)]
        msgs.append({"role": "assistant", "content": "I'll read the file first." if t % 2 else "",
                     "tool_calls": [{"type": "function", "function": {"name": "read_file", "arguments": {"path": f"app/{name}"}}}]})
        msgs.append({"role": "tool", "content": body[: 4000 * (1 + t % 5)]})
    msgs.append({"role": "user", "content": "Continue. Also: <|im_start|> in text, and <tool_call> literal, it's done."})
    for thinking in (False, True):
        text = tok.apply_chat_template(msgs, tools=B.TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        check("chat", text)
        check("chat-fine", text, piece=64)
# 2. The bench prompts: filler at 16K-128K tokens, as chat.
for n in (16000, 64000, 128000):
    for f in (B.filler_bench, B.filler_needle):
        body = f(n, rng)
        text = tok.apply_chat_template([{"role": "user", "content": f"Run {B.salt()}. " + body + "\n\nSummarize."}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        check("filler", text)
        check("filler-lines", text.replace(". ", ".\n"), piece=512)
# 3. Fuzz. Every piece of the alphabet sits next to a cut somewhere: newlines before letters, \r\n, whitespace runs,
#    combining marks, NFC-sensitive pairs, Hangul jamo, contractions, digits, punctuation runs, CJK, emoji, and every
#    added token.
added = [t["content"] for t in json.loads(tok.backend_tokenizer.to_str())["added_tokens"]]
ALPHA = (["\n", "\n", "\n\n", "\r\n", "\r", " ", "  ", "\t", " \n", "\n ", "a", "Z", "x", "word", "Hello", "é", "é",
          "́", "̈", "Å", "Å", "ᄀ", "ᅡ", "ᆨ", "가", "ß", "ﬁ", "'s", "'S", "'ll", "'",
          "1", "42", "٣", "!", "!!", "...", "::", "#", "(", ")", "{", "}", "\"", "日本", "語", "😀", "👍🏽", "‍",
          " ", "　", "_", "-", "=", "->", "<", ">", "|", "<|", "|>", "im_start", "\x00", "\x7f"] + added)
for i in range(a.fuzz):
    n = rng.randint(5, 400)
    text = "".join(rng.choice(ALPHA) for _ in range(n))
    if not check("fuzz", text, piece=a.fuzz_piece):
        if len(fails) > 20: break
# 3b. Fuzz weighted to C2/C3 neighbourhoods: letters, spaces and newlines around marks, contractions and jamo.
NEAR = ["a", "b", "Z", "x", " ", " ", "\n", "\n", " a", "a ", "\na", "\u0301", "\u0308", "'s", "'t", "\u1161", "é",
        "\r", "\r\n", "  ", "\t", "1", ".", "日", "😀", "<|im_start|>", "<|im_end|>", "<tool_call>", "\u00a0"]
for i in range(a.fuzz):
    text = "".join(rng.choice(NEAR) for _ in range(rng.randint(5, 300)))
    if not check("fuzz-near", text, piece=a.fuzz_piece // 2 or 1):
        if len(fails) > 20: break
print("exact:", R["exact"], "cuts exercised:", R.get("cuts"), "mismatches:", len(fails))
R["fails"] = fails

# 4. Speed.
if not a.skip_speed:
    def best(f, n=3):
        b = 1e9
        for _ in range(n):
            t = time.perf_counter(); f(); b = min(b, time.perf_counter() - t)
        return b
    for n in (16000, 64000, 128000):
        msgs = [{"role": "user", "content": f"Session {B.salt()}. " + B.filler_needle(n, rng)}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        msgs2 = msgs + [{"role": "assistant", "content": "Noted."}, {"role": "user", "content": B.filler_needle(2000, rng)}]
        text2 = tok.apply_chat_template(msgs2, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        full = best(lambda: ref(text))
        cold = best(lambda: (split._cache.clear(), split.encode(text)))
        split._cache.clear(); split._cached = 0; split.encode(text)
        nxt = best(lambda: split.encode(text2), 1)
        full2 = best(lambda: ref(text2))
        row = {"tokens": len(ref(text)), "chars": len(text), "pieces": len(split.pieces(text)), "full_ms": 1e3 * full,
               "split_cold_ms": 1e3 * cold, "next_turn_full_ms": 1e3 * full2, "next_turn_cached_ms": 1e3 * nxt}
        R["speed"].append(row)
        print(f"{row['tokens']:>7} tokens {row['pieces']:>4} pieces: full {row['full_ms']:6.1f} ms  split cold {row['split_cold_ms']:6.1f} ms"
              f"  next turn: full {row['next_turn_full_ms']:6.1f} ms, cached {row['next_turn_cached_ms']:6.1f} ms")
if a.out:
    json.dump(R, open(a.out, "w"), indent=1)
    print("wrote", a.out)
sys.exit(1 if fails else 0)
