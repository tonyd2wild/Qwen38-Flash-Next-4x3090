# SPDX-License-Identifier: Apache-2.0
# 2026-10-17 (T11, journal/14-p9-fixed-overheads.md). Exact split tokenization for long prompts (TOKSPLIT=1). A 128K
# chat prompt costs ~0.3 s of single-threaded Rust tokenization inside the server's TTFT, on every turn. The text is
# cut at positions where tokenizing the pieces separately provably gives the same ids as tokenizing the whole:
#   C1  just before <|im_start|> (an added-token boundary: segments around added tokens are encoded independently)
#   C2  between "\n" and an ASCII letter (no added token contains "\n"; "\n" never composes under NFC; the regex match
#       that covers the "\n" is \s*[\r\n]+ or ' ?[^\s...]+[\r\n]*', both end at the letter, and no match to its left
#       looks past it)
#   C3  between an ASCII letter and a space that is followed by an ASCII letter ("word| word"; no added token contains a
#       space; the letter run ends at the space under every alternative, and " word" starts the next match)
# The argument holds only for the tokenizer pipeline checked in _unsupported(); anything else disables the split.
# Pieces are cached by their text (an agent's next turn repeats the previous turn's pieces), and the misses are encoded
# in one encode_batch_fast call (rayon threads; RAYON_NUM_THREADS caps them).
"""Exact piece-wise tokenization with a piece cache."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from array import array
from collections import OrderedDict
from collections.abc import Callable

from vllm.logger import init_logger

logger = init_logger(__name__)

# The Qwen pre-tokenizer regex as tokenizer.json has it, and the variant transformers builds from tokenizer_config.json
# (pretokenize_regex: \p{M} added inside the letter and symbol classes). The C2 argument holds for both.
_REGEXES = {
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
}
_IM_START = "<|im_start|>"
# The cut is at m.end(): after the "\n" (C2), or at the zero-width position before the space (C3).
_CUT = re.compile(r"\n(?=[A-Za-z])|(?<=[A-Za-z])(?= [A-Za-z])")
_ENCODE_KEYS = {"truncation", "max_length", "add_special_tokens"}


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _unsupported(cfg: dict) -> str | None:
    """Why the C1/C2 argument does not apply to this tokenizer config (None if it does)."""
    norm = cfg.get("normalizer")
    if norm is not None and norm.get("type") != "NFC":
        return f"normalizer {norm.get('type')}"
    pre = cfg.get("pre_tokenizer") or {}
    seq = pre.get("pretokenizers") if pre.get("type") == "Sequence" else None
    if not seq or len(seq) != 2:
        return f"pre-tokenizer {json.dumps(pre)[:200]}"
    split, bl = seq
    if (split.get("type") != "Split" or split.get("behavior") != "Isolated" or split.get("invert")
            or (split.get("pattern") or {}).get("Regex") not in _REGEXES):
        return f"split pre-tokenizer {json.dumps(split)[:200]}"
    if bl.get("type") != "ByteLevel" or bl.get("use_regex") or bl.get("add_prefix_space"):
        return f"second pre-tokenizer {json.dumps(bl)[:200]}"
    model = cfg.get("model") or {}
    if model.get("type") != "BPE" or model.get("dropout"):
        return f"model {model.get('type')} dropout {model.get('dropout')}"
    added = [a["content"] for a in cfg.get("added_tokens", [])]
    for a in cfg.get("added_tokens", []):
        if a.get("lstrip") or a.get("rstrip") or a.get("single_word") or a.get("normalized"):
            return f"added token {a['content']!r} has strip/single_word/normalized flags"
        if "\n" in a["content"] or " " in a["content"]:
            return f"added token {a['content']!r} contains a newline or a space"
    if _IM_START not in added:
        return f"no {_IM_START} added token"
    for a in added:  # no other added token may overlap an occurrence of <|im_start|>
        if a != _IM_START and _IM_START in a:
            return f"added token {a!r} contains {_IM_START}"
        for k in range(1, len(_IM_START)):
            if a.endswith(_IM_START[:k]) or (a != _IM_START and a.startswith(_IM_START[-k:])):
                return f"added token {a!r} overlaps {_IM_START}"
    return None


class TokSplit:
    """Token ids of long prompts, computed piece by piece and exactly equal to the tokenizer's own call."""

    def __init__(self, backend, reference: Callable[[str, bool], list[int]], *, min_piece: int = 4096,
                 min_text: int = 16384, cache_tokens: int = 4_000_000) -> None:
        from tokenizers import Tokenizer

        cfg_str = backend.to_str()
        why = _unsupported(json.loads(cfg_str))
        if why:
            raise ValueError(why)
        # A private copy: transformers' __call__ sets truncation on the shared backend and leaves it set.
        self._tok = Tokenizer.from_str(cfg_str)
        self._tok.no_truncation()
        self._tok.no_padding()
        self.min_piece, self.min_text, self.cache_tokens = min_piece, min_text, cache_tokens
        self._cache: OrderedDict[str, array] = OrderedDict()
        self._cached = 0
        self._lock = threading.Lock()
        self.calls = self.hit_tokens = self.miss_tokens = 0
        self.seconds = 0.0
        self._self_test(reference, [a["content"] for a in json.loads(cfg_str)["added_tokens"]])

    def _self_test(self, reference, added: list[str]) -> None:
        # Both add_special_tokens settings must equal the plain piece concatenation (no post-processor tokens), on a
        # sample with every added token and every cut kind, cut far finer than in service.
        words = ["def f(x):", "  return x", "Hello world", "naïve", "it's here", "é", "日本語", "😀", "it's", "x = 1\r\n",
                 "\t", "  ", "123", "!!", "<|im_start|>", "über"]
        parts = []
        for i in range(600):
            parts.append(words[i % len(words)])
            parts.append(added[i % len(added)] if i % 7 == 0 else "")
            parts.append("\n" if i % 3 else " ")
        sample = "".join(parts) + "\nEnd"
        mp, mt = self.min_piece, self.min_text
        self.min_piece, self.min_text = 16, 0
        try:
            if len(self.pieces(sample)) < 50:
                raise ValueError("self-test: too few pieces")
            got = self.encode(sample, use_cache=False)
        finally:
            self.min_piece, self.min_text = mp, mt
        for ast in (False, True):
            want = list(reference(sample, ast))
            if got != want:
                n = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
                raise ValueError(f"self-test mismatch (add_special_tokens={ast}) at token {n} of {len(want)}")

    def pieces(self, text: str) -> list[str]:
        """Cut text at every C1 point and, inside a message, at the first C2/C3 point min_piece characters on."""
        out, bounds, i = [], [0], text.find(_IM_START, 1)
        while i > 0:
            bounds.append(i)
            i = text.find(_IM_START, i + 1)
        bounds.append(len(text))
        for a, b in zip(bounds, bounds[1:]):
            last = a
            while b - last > self.min_piece:
                m = _CUT.search(text, last + self.min_piece, b)
                if m is None:
                    break
                out.append(text[last:m.end()])
                last = m.end()
            if b > last:
                out.append(text[last:b])
        return out

    def encode(self, text: str, use_cache: bool = True) -> list[int]:
        t0 = time.perf_counter()
        ps = self.pieces(text)
        with self._lock:
            got = [self._cache.get(p) if use_cache else None for p in ps]
            for p, g in zip(ps, got):
                if g is not None:
                    self._cache.move_to_end(p)
        miss = [i for i, g in enumerate(got) if g is None]
        if miss:
            for i, e in zip(miss, self._tok.encode_batch_fast([ps[i] for i in miss], add_special_tokens=False)):
                got[i] = array("i", e.ids)
        ids: list[int] = []
        for g in got:
            ids.extend(g)
        if use_cache:
            with self._lock:
                for i in miss:
                    if ps[i] not in self._cache:
                        self._cache[ps[i]] = got[i]
                        self._cached += len(got[i])
                while self._cached > self.cache_tokens and self._cache:
                    self._cached -= len(self._cache.popitem(last=False)[1])
                n_miss = sum(len(got[i]) for i in miss)
                self.calls += 1
                self.miss_tokens += n_miss
                self.hit_tokens += len(ids) - n_miss
                self.seconds += time.perf_counter() - t0
                if self.calls == 1 or self.calls % 32 == 0:
                    logger.info("TOKSPLIT: %d calls, %d tokens from the piece cache, %d encoded, %.3f s in total; "
                                "last call %d tokens in %d pieces (%d encoded) in %.1f ms; cache %d tokens", self.calls,
                                self.hit_tokens, self.miss_tokens, self.seconds, len(ids), len(ps), n_miss,
                                1e3 * (time.perf_counter() - t0), self._cached)
        return ids

    def maybe_encode(self, text: str, kwargs: dict) -> list[int] | None:
        """The ids tokenizer(text, **kwargs) would return, or None to make the caller use that call."""
        if len(text) < self.min_text or not set(kwargs) <= _ENCODE_KEYS:
            return None
        ids = self.encode(text)
        if kwargs.get("truncation") and kwargs.get("max_length") is not None and len(ids) > kwargs["max_length"]:
            return None  # over-length: the original call truncates exactly as before (the request then fails)
        return ids


_INSTANCES: dict[object, TokSplit | None] = {}


def make_tok_split(tokenizer) -> TokSplit | None:
    """One TokSplit per tokenizer (None, with a log line, when the split does not apply)."""
    key = getattr(tokenizer, "name_or_path", None) or id(tokenizer)
    if key in _INSTANCES:
        return _INSTANCES[key]
    split = None
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        logger.warning("TOKSPLIT disabled: %s has no Rust backend tokenizer", type(tokenizer).__name__)
    else:
        try:
            t0 = time.perf_counter()
            split = TokSplit(backend, lambda t, ast: tokenizer(t, add_special_tokens=ast)["input_ids"],
                             min_piece=_env_int("QWEN4EXP_TOKSPLIT_MIN_PIECE", 4096),
                             min_text=_env_int("QWEN4EXP_TOKSPLIT_MIN_TEXT", 16384),
                             cache_tokens=_env_int("QWEN4EXP_TOKSPLIT_CACHE_TOKENS", 4_000_000))
            logger.info("TOKSPLIT on: pieces of >= %d chars for prompts of >= %d chars, cache %d tokens, "
                        "RAYON_NUM_THREADS=%s, self-test passed in %.2f s", split.min_piece, split.min_text,
                        split.cache_tokens, os.environ.get("RAYON_NUM_THREADS", "unset"), time.perf_counter() - t0)
        except ValueError as e:
            logger.warning("TOKSPLIT disabled: %s", e)
    _INSTANCES[key] = split
    return split


__all__ = ["TokSplit", "make_tok_split"]
