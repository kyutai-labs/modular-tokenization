"""Pretraining dataloader: monolingual batches with sampled subtokenizers.

Each batch contains documents of ONE language, tokenized with ONE
subtokenizer (the language's own, or a sampled unified one — paper §4). Token
ids are in the global id space; the batch carries the subtokenizer's
logits mask so the loss can be restricted to its sub-vocabulary.

Randomness is stateless-by-construction so that resume is bit-exact with a
tiny saved state (paper-repo redesign):

  - one seed chain per (rank, language) samples each DOCUMENT's subtokenizer
    (advanced once per document read: seed <- next_seed(seed));
  - one seed chain per rank samples the LANGUAGE of each batch
    (advanced once per emitted batch);
  - chains derive from the training seed (domain constants keep them
    independent) and never reset, so every epoch re-rolls naturally and
    different seeds give different samplings;
  - a seed is consumed as ``random.Random(seed).choices(...)`` — a pure
    function of the seed. NOTE: Python only guarantees the seed->output
    mapping of ``choices`` within a Python version, so bit-exact resume
    assumes save and resume run the same Python (recorded in checkpoint meta).

The snapshot therefore stores no token buffers: per language, the reader head
and the position + seed of the earliest document with unconsumed
tokens (the *anchor*); per buffer, how much of its documents was already
consumed. ``restore`` replays the pipeline from the anchor — re-rolling the same
seed chain, re-tokenizing a handful of documents — and drops what was already
consumed, rebuilding the buffers bit-identically.

Sources are explicit user paths ({lang: path}); a path may be a JSONL file, a
.zst-compressed JSONL file, a plain text file (one document per line), or a
directory of such files. The language mixture is explicit ({lang: weight}).
"""
import json
import math
import os
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from modular_lm.data.utils import create_seed, iter_file_lines, next_seed, source_files

# Domain constants: keep the two seed-chain families independent even for the
# same training seed (see data.utils.create_seed).
LANGUAGE_SAMPLING, SUBTOKENIZER_SAMPLING = 1, 2


# ---------------------------------------------------------------------------
# Config and batch
# ---------------------------------------------------------------------------

class PretrainDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: dict[str, str] = {}   # {lang: path} — file (.jsonl/.zst/.txt) or directory
    weights: dict[str, float] = {}  # {lang: mixture weight} — explicit (normalized here)
    text_field: str = "text"        # document field in JSONL sources
    batch_size: int | None = None
    context_size: int | None = None
    seed: int = 0
    # read-ahead depth: a buffer pre-reads up to this many batches' worth of
    # tokens before emitting (bounds the resume replay window).
    buffer_coef: int = 4

    # deterministic document filters (None = disabled). Metadata filters read
    # the corresponding JSONL fields when present.
    min_doc_chars: int | None = None
    max_doc_chars: int | None = None
    filter_lid: float | None = None                # keep if doc['lid'] >= value
    threshold_repetitions: float | None = None     # keep if doc['repetitions'] <= value
    threshold_long_words: float | None = None      # keep if doc['long_words'] <= value
    quality_weights: str | None = None             # 'wiki:1.0,stem:1.0,...'
    quality_threshold: float | None = None         # keep if weighted score >= value

    @model_validator(mode="after")
    def validate_weights(self):
        if self.sources and self.weights:
            if set(self.weights) != set(self.sources):
                raise ValueError("weights and sources must have the same languages")
        return self


@dataclass
class Batch:
    x: np.ndarray                 # (bsz, csz) inputs, global token ids
    y: np.ndarray                 # (bsz, csz) targets (inputs shifted by one)
    logits_mask: np.ndarray       # (bsz, vocab) additive mask: 0 inside the subtokenizer's vocab, -inf outside
    c_bpb: float                  # tokens-per-byte / ln 2 (loss -> bits-per-byte factor)
    lang: str
    subtokenizer_id: str


def make_example_filter(cfg: PretrainDataConfig):
    """Deterministic document filter (a pure function of the document — any
    randomness here would break replay-based resume)."""
    coefs = None
    if cfg.quality_weights:
        coefs = {k: float(v) for k, v in (x.split(":") for x in cfg.quality_weights.split(","))}

    def _filter(doc: dict, text: str) -> bool:
        if cfg.min_doc_chars is not None and len(text) < cfg.min_doc_chars:
            return False
        if cfg.max_doc_chars is not None and len(text) > cfg.max_doc_chars:
            return False
        if cfg.filter_lid is not None and "lid" in doc and doc["lid"] < cfg.filter_lid:
            return False
        if cfg.threshold_repetitions is not None and "repetitions" in doc \
                and doc["repetitions"] > cfg.threshold_repetitions:
            return False
        if cfg.threshold_long_words is not None and "long_words" in doc \
                and doc["long_words"] > cfg.threshold_long_words:
            return False
        if coefs is not None and cfg.quality_threshold is not None:
            if any(k in doc for k in coefs):
                score = sum(v * doc.get(k, 0.0) for k, v in coefs.items())
                if score < cfg.quality_threshold:
                    return False
        return True

    return _filter


# ---------------------------------------------------------------------------
# Document reading (any source layout; per-rank sharding by line number)
# ---------------------------------------------------------------------------

class DocReader:
    """One language's document stream, owning its read position.

    Yields ``(text, pos)`` where ``pos = (wrap, file_id, line_id)`` is the
    document's stable address (wrap = how many times the source was cycled).
    Each rank reads the disjoint ``line_id % world_size == rank`` slice.
    """

    def __init__(self, path: str, text_field: str, example_filter, rank: int, world_size: int):
        self.files = source_files(path)
        self.text_field = text_field
        self.example_filter = example_filter
        self.rank = rank
        self.world_size = world_size
        # position of the next line to read
        self.wrap = 0
        self.file_id = 0
        self.line_id = 0
        self._it = None

    def _parse(self, path: str, line: str):
        if path.endswith(".txt"):
            text = line.rstrip("\n")
            return ({}, text) if text.strip() else None
        doc = json.loads(line)
        return doc, doc[self.text_field]

    def _lines(self):
        """Infinite (doc, text, pos) stream from the current position."""
        while True:
            while self.file_id < len(self.files):
                path = self.files[self.file_id]
                for line_id, line in enumerate(iter_file_lines(path)):
                    if line_id < self.line_id:
                        continue
                    self.line_id = line_id + 1
                    if line_id % self.world_size != self.rank:
                        continue
                    parsed = self._parse(path, line)
                    if parsed is None:
                        continue
                    doc, text = parsed
                    if self.example_filter(doc, text):
                        yield text, (self.wrap, self.file_id, line_id)
                self.file_id += 1
                self.line_id = 0
            self.wrap += 1
            self.file_id = 0

    def next_doc(self):
        if self._it is None:
            self._it = self._lines()
        return next(self._it)

    def head(self) -> tuple:
        return (self.wrap, self.file_id, self.line_id)

    def seek(self, pos: tuple):
        self.wrap, self.file_id, self.line_id = pos
        self._it = None


# ---------------------------------------------------------------------------
# Buffers: one token buffer per (language, subtokenizer)
# ---------------------------------------------------------------------------

@dataclass
class _BufferedDoc:
    """One tokenized document awaiting consumption in a subtokenizer buffer —
    together with what resume needs to rebuild it: its address, and the
    subtokenizer seed as it was before this document's draw (saving it per
    document is what makes the resume anchor free to compute)."""
    pos: tuple
    seed_before: int              # the chain restart point for resume replay
    tokens: np.ndarray
    n_bytes: int
    n_consumed: int = 0           # tokens of this document already consumed


class SubtokenizerBuffer:
    def __init__(self, subtokenizer_id: str, logits_mask: np.ndarray, n_target: int):
        self.subtokenizer_id = subtokenizer_id
        self.logits_mask = logits_mask
        self.n_target = n_target
        self.docs: deque[_BufferedDoc] = deque()
        self.n_unconsumed = 0
        # where consumption stopped: address of the last consumed doc + tokens taken from it
        self.last_consumed_pos: tuple | None = None
        self.last_consumed_n_tokens: int = 0

    def push(self, doc: _BufferedDoc):
        self.docs.append(doc)
        self.n_unconsumed += len(doc.tokens) - doc.n_consumed

    def ready(self) -> bool:
        return self.n_unconsumed >= self.n_target

    def take_batch_tokens(self) -> tuple[np.ndarray, float]:
        """Consume exactly ``n_target`` tokens (one batch's worth) from the
        buffered-document FIFO, popping documents as they finish, and return
        them flat together with the batch's bits-per-byte factor."""
        out, taken = [], 0
        n_tokens = n_bytes = 0
        while taken < self.n_target:
            doc = self.docs[0]
            need = self.n_target - taken
            grab = min(need, len(doc.tokens) - doc.n_consumed)
            out.append(doc.tokens[doc.n_consumed:doc.n_consumed + grab])
            frac = grab / len(doc.tokens)
            n_tokens += grab
            n_bytes += doc.n_bytes * frac
            doc.n_consumed += grab
            taken += grab
            self.last_consumed_pos, self.last_consumed_n_tokens = doc.pos, doc.n_consumed
            if doc.n_consumed == len(doc.tokens):
                self.docs.popleft()
        self.n_unconsumed -= self.n_target
        c_bpb = (n_tokens / n_bytes / math.log(2)) if n_bytes > 0 else 0.0
        return np.concatenate(out), c_bpb


# ---------------------------------------------------------------------------
# The dataset
# ---------------------------------------------------------------------------

class PretrainDataset:
    """Iterable of Batch, with bit-exact ``snapshot()``/``restore()``.

    ``tokenizer`` is a ``modular_tokenizers`` ModularTokenizer (subtokenizers
    from its sampling strategy) or a plain Tokenizer (single 'all' buffer).
    """

    def __init__(self, config: PretrainDataConfig, tokenizer, rank: int = 0, world_size: int = 1):
        assert config.sources, "sources must map languages to paths"
        assert config.batch_size and config.context_size, "batch_size and context_size are required"
        self.config = config
        self.tokenizer = tokenizer
        self.rank = rank
        self.world_size = world_size
        self.is_modular = hasattr(tokenizer, "subtokenizers")

        self.langs = sorted(config.sources)
        weights = config.weights or {l: 1.0 for l in self.langs}
        self.lang_weights = [weights[l] for l in self.langs]

        example_filter = make_example_filter(config)
        n_target = config.batch_size * (config.context_size + 1)

        self.doc_readers: dict[str, DocReader] = {}
        self.buffers: dict[str, dict[str, SubtokenizerBuffer]] = {}
        self.subtokenizer_ids: dict[str, list[str]] = {}
        self.subtokenizer_probs: dict[str, list[float]] = {}
        # one seed chain per language: samples the subtokenizer of each document
        self.subtokenizer_seeds: dict[str, int] = {}
        for lang in self.langs:
            self._add_language(lang, config, tokenizer, example_filter, n_target, rank, world_size)

        self.language_seed = create_seed(config.seed, LANGUAGE_SAMPLING, rank)

    def _add_language(self, lang, config, tokenizer, example_filter, n_target, rank, world_size):
        """Create one language's document reader, subtokenizer sampling table,
        buffers, and seed chain."""
        self.doc_readers[lang] = DocReader(
            config.sources[lang], config.text_field, example_filter, rank, world_size
        )
        if self.is_modular:
            ids = tokenizer.subtokenizer_ids_by_lang[lang]
            probs = tokenizer.subtokenizer_probs.get(lang) if tokenizer.subtokenizer_probs else None
            probs = [probs[i] for i in ids] if probs else [1.0] * len(ids)
        else:
            ids, probs = ["all"], [1.0]
        self.subtokenizer_ids[lang] = ids
        self.subtokenizer_probs[lang] = probs
        self.buffers[lang] = {
            i: SubtokenizerBuffer(i, self._logits_mask(i), n_target) for i in ids
        }
        self.subtokenizer_seeds[lang] = create_seed(config.seed, SUBTOKENIZER_SAMPLING, rank, lang)

    # -- masks / tokenization ------------------------------------------------

    def _logits_mask(self, subtokenizer_id: str) -> np.ndarray:
        bsz = self.config.batch_size
        if self.is_modular:
            mask_bool = self.tokenizer.mask(subtokenizer_id)
            mask = np.full((1, len(mask_bool)), -np.inf, dtype=np.float32)
            mask[0, mask_bool] = 0.0
            return np.repeat(mask, bsz, axis=0)
        return np.zeros((bsz, self.tokenizer.vocab_size()), dtype=np.float32)

    def _tokenize(self, text: str, subtokenizer_id: str) -> np.ndarray:
        if self.is_modular:
            return np.asarray(self.tokenizer.encode(text, subtokenizer_id), dtype=np.int32)
        return np.asarray(self.tokenizer.encode(text), dtype=np.int32)

    # -- subtokenizer sampling (one seed chain per language) ----------------

    def _buffer_next_doc(self, lang: str):
        """Read the language's next document, sample its subtokenizer, and
        tokenize it into that subtokenizer's buffer."""
        text, pos = self.doc_readers[lang].next_doc()
        seed_before = self.subtokenizer_seeds[lang]
        self.subtokenizer_seeds[lang] = next_seed(seed_before)
        ids, probs = self.subtokenizer_ids[lang], self.subtokenizer_probs[lang]
        sub_id = random.Random(self.subtokenizer_seeds[lang]).choices(ids, weights=probs)[0]
        buffer = self.buffers[lang][sub_id]

        last_consumed = (buffer.last_consumed_pos, buffer.last_consumed_n_tokens)
        if buffer.last_consumed_pos is not None and pos < buffer.last_consumed_pos:
            return  # replay: this document was already fully consumed
        tokens = self._tokenize(text, sub_id)
        doc = _BufferedDoc(pos, seed_before, tokens, len(text.encode("utf-8")))
        if last_consumed[0] is not None and pos == last_consumed[0]:
            doc.n_consumed = min(last_consumed[1], len(tokens))  # replay: partially consumed
        if doc.n_consumed < len(doc.tokens):
            buffer.push(doc)

    # -- batch iteration (the language seed chain) --------------------------

    def __iter__(self):
        return self

    def __next__(self) -> Batch:
        self.language_seed = next_seed(self.language_seed)
        lang = random.Random(self.language_seed).choices(self.langs, weights=self.lang_weights)[0]

        buffers = self.buffers[lang]
        while not any(b.ready() for b in buffers.values()):
            self._buffer_next_doc(lang)
        buffer = next(b for b in buffers.values() if b.ready())

        tokens, c_bpb = buffer.take_batch_tokens()
        csz1 = self.config.context_size + 1
        tokens = tokens.reshape(self.config.batch_size, csz1)
        return Batch(
            x=tokens[:, :-1],
            y=tokens[:, 1:],
            logits_mask=buffer.logits_mask,
            c_bpb=c_bpb,
            lang=lang,
            subtokenizer_id=buffer.subtokenizer_id,
        )

    # -- snapshot / restore ---------------------------------------------------

    def snapshot(self) -> dict:
        """JSON-serializable resume state: chain anchors and consumption
        boundaries — never token buffers."""
        langs_state = {}
        for lang in self.langs:
            # anchor = earliest buffered document with unconsumed tokens
            anchor = None
            for s in self.buffers[lang].values():
                for doc in s.docs:
                    if doc.n_consumed < len(doc.tokens) and (anchor is None or doc.pos < anchor[0]):
                        anchor = (doc.pos, doc.seed_before)
            head = self.doc_readers[lang].head()
            langs_state[lang] = {
                "head": list(head),
                "anchor_pos": list(anchor[0]) if anchor else list(head),
                "anchor_seed": anchor[1] if anchor else self.subtokenizer_seeds[lang],
                "buffers": {
                    c: {
                        "last_consumed_pos": list(s.last_consumed_pos) if s.last_consumed_pos else None,
                        "last_consumed_n_tokens": s.last_consumed_n_tokens,
                    }
                    for c, s in self.buffers[lang].items()
                },
            }
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "seed": self.config.seed,
            "language_seed": self.language_seed,
            "langs": langs_state,
        }

    def restore(self, snap: dict):
        assert snap["world_size"] == self.world_size, (
            f"resume world_size mismatch: checkpoint={snap['world_size']} current={self.world_size}"
        )
        assert snap["rank"] == self.rank, "resume rank mismatch"
        assert snap["seed"] == self.config.seed, "resume seed mismatch"
        assert set(snap["langs"]) == set(self.langs), "resume language set mismatch"

        self.language_seed = snap["language_seed"]
        for lang, st in snap["langs"].items():
            reader = self.doc_readers[lang]
            head = tuple(st["head"])
            anchor_pos = tuple(st["anchor_pos"])

            assert set(st["buffers"]) == set(self.buffers[lang]), (
                f"resume subtokenizer set mismatch for '{lang}' — the tokenizer must be "
                "rebuilt identically (same seed) before restoring"
            )
            # restore consumption boundaries so replay drops what was consumed
            for c, b in st["buffers"].items():
                buffer = self.buffers[lang][c]
                buffer.last_consumed_pos = tuple(b["last_consumed_pos"]) if b["last_consumed_pos"] else None
                buffer.last_consumed_n_tokens = b["last_consumed_n_tokens"]

            # replay from the anchor: re-roll the chain, re-tokenize, refill
            reader.seek(anchor_pos)
            self.subtokenizer_seeds[lang] = st["anchor_seed"]
            while reader.head() < head:
                self._buffer_next_doc(lang)
            assert reader.head() == head, (
                f"replay overshoot for '{lang}': {reader.head()} != {head} "
                "(corpus changed since the checkpoint?)"
            )
