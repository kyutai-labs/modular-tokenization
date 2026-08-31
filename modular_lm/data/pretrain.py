"""Pretraining dataloader: monolingual batches with sampled subtokenizers.

Each batch contains documents of ONE language, tokenized with ONE composition
(the language's own subtokenizer, or a sampled combination — paper §4). Token
ids are in the global id space; the batch carries the composition's bias mask
so the loss can be restricted to its sub-vocabulary.

Randomness is stateless-by-construction so that resume is bit-exact with a
tiny saved state (paper-repo redesign):

  - one seed chain per (rank, language) samples each DOCUMENT's composition
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
tokens (the *anchor*); per stream, how much of its documents was already
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

from modular_lm.data.utils import derive_seed, iter_file_lines, next_seed, source_files

# Domain constants: keep the two seed-chain families independent even for the
# same training seed (see utils.seeds.derive_seed).
LANGUAGE_SAMPLING, COMPOSITION_SAMPLING = 1, 2


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
    # read-ahead depth: a stream pre-buffers up to this many chunks' worth of
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
    masked_tokens: np.ndarray     # (bsz, vocab) bias: 0 inside the composition, -inf outside
    c_bpb: float                  # tokens-per-byte / ln 2 (loss -> bits-per-byte factor)
    lang: str
    composition_id: str


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
# Streams: one buffered token stream per (language, composition)
# ---------------------------------------------------------------------------

@dataclass
class _Segment:
    """One buffered document: its tokens, address, and the composition seed
    as it was BEFORE this document's draw (saving it per segment is what makes
    the resume anchor free to compute)."""
    pos: tuple
    seed_pred: int
    tokens: np.ndarray
    n_bytes: int
    off: int = 0                  # tokens already consumed


class Stream:
    def __init__(self, composition_id: str, bias_mask: np.ndarray, n_target: int):
        self.composition_id = composition_id
        self.bias_mask = bias_mask
        self.n_target = n_target
        self.segments: deque[_Segment] = deque()
        self.available = 0
        # consumption boundary: address + token offset of the last consumed doc
        self.boundary_pos: tuple | None = None
        self.boundary_off: int = 0

    def push(self, seg: _Segment):
        self.segments.append(seg)
        self.available += len(seg.tokens) - seg.off

    def ready(self) -> bool:
        return self.available >= self.n_target

    def take_chunk(self) -> tuple[np.ndarray, float]:
        """Consume exactly n_target tokens from the segment FIFO."""
        out, taken = [], 0
        n_tokens = n_bytes = 0
        while taken < self.n_target:
            seg = self.segments[0]
            need = self.n_target - taken
            grab = min(need, len(seg.tokens) - seg.off)
            out.append(seg.tokens[seg.off:seg.off + grab])
            frac = grab / len(seg.tokens)
            n_tokens += grab
            n_bytes += seg.n_bytes * frac
            seg.off += grab
            taken += grab
            self.boundary_pos, self.boundary_off = seg.pos, seg.off
            if seg.off == len(seg.tokens):
                self.segments.popleft()
        self.available -= self.n_target
        c_bpb = (n_tokens / n_bytes / math.log(2)) if n_bytes > 0 else 0.0
        return np.concatenate(out), c_bpb


# ---------------------------------------------------------------------------
# The dataset
# ---------------------------------------------------------------------------

class PretrainDataset:
    """Iterable of Batch, with bit-exact ``snapshot()``/``restore()``.

    ``tokenizer`` is a ``modular_tokenizers`` ModularTokenizer (compositions
    from its sampling strategy) or a plain Tokenizer (single 'all' stream).
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

        self.readers: dict[str, DocReader] = {}
        self.streams: dict[str, dict[str, Stream]] = {}
        self.route: dict[str, tuple[list[str], list[float]]] = {}
        # one seed chain per language: samples the composition of each document
        self.composition_seeds: dict[str, int] = {}
        for lang in self.langs:
            self.readers[lang] = DocReader(
                config.sources[lang], config.text_field, example_filter, rank, world_size
            )
            if self.is_modular:
                comps = tokenizer.compositions_by_lang[lang]
                probs = tokenizer.composition_probs.get(lang) if tokenizer.composition_probs else None
                probs = [probs[c] for c in comps] if probs else [1.0] * len(comps)
            else:
                comps, probs = ["all"], [1.0]
            self.route[lang] = (comps, probs)
            self.streams[lang] = {
                c: Stream(c, self._bias_mask(c), n_target) for c in comps
            }
            self.composition_seeds[lang] = derive_seed(config.seed, COMPOSITION_SAMPLING, rank, lang)

        self.language_seed = derive_seed(config.seed, LANGUAGE_SAMPLING, rank)

    # -- masks / tokenization ------------------------------------------------

    def _bias_mask(self, composition_id: str) -> np.ndarray:
        bsz = self.config.batch_size
        if self.is_modular:
            mask_bool = self.tokenizer.mask(composition_id)
            mask = np.full((1, len(mask_bool)), -np.inf, dtype=np.float32)
            mask[0, mask_bool] = 0.0
            return np.repeat(mask, bsz, axis=0)
        return np.zeros((bsz, self.tokenizer.vocab_size()), dtype=np.float32)

    def _tokenize(self, text: str, composition_id: str) -> np.ndarray:
        if self.is_modular:
            return np.asarray(self.tokenizer.encode(text, composition_id), dtype=np.int32)
        return np.asarray(self.tokenizer.encode(text), dtype=np.int32)

    # -- composition sampling (one seed chain per language) ----------------

    def _route_next_doc(self, lang: str):
        """Read one document, roll the language's chain, tokenize into the
        chosen composition's stream."""
        text, pos = self.readers[lang].next_doc()
        seed_pred = self.composition_seeds[lang]
        self.composition_seeds[lang] = next_seed(seed_pred)
        comps, probs = self.route[lang]
        comp = random.Random(self.composition_seeds[lang]).choices(comps, weights=probs)[0]
        stream = self.streams[lang][comp]

        boundary = (stream.boundary_pos, stream.boundary_off)
        if stream.boundary_pos is not None and pos < stream.boundary_pos:
            return  # replay: this document was already fully consumed
        tokens = self._tokenize(text, comp)
        seg = _Segment(pos, seed_pred, tokens, len(text.encode("utf-8")))
        if boundary[0] is not None and pos == boundary[0]:
            seg.off = min(boundary[1], len(tokens))  # replay: partially consumed
        if seg.off < len(seg.tokens):
            stream.push(seg)

    # -- batch iteration (the language seed chain) --------------------------

    def __iter__(self):
        return self

    def __next__(self) -> Batch:
        self.language_seed = next_seed(self.language_seed)
        lang = random.Random(self.language_seed).choices(self.langs, weights=self.lang_weights)[0]

        streams = self.streams[lang]
        while not any(s.ready() for s in streams.values()):
            self._route_next_doc(lang)
        stream = next(s for s in streams.values() if s.ready())

        chunk, c_bpb = stream.take_chunk()
        csz1 = self.config.context_size + 1
        chunk = chunk.reshape(self.config.batch_size, csz1)
        return Batch(
            x=chunk[:, :-1],
            y=chunk[:, 1:],
            masked_tokens=stream.bias_mask,
            c_bpb=c_bpb,
            lang=lang,
            composition_id=stream.composition_id,
        )

    # -- snapshot / restore ---------------------------------------------------

    def snapshot(self) -> dict:
        """JSON-serializable resume state: chain anchors and consumption
        boundaries — never token buffers."""
        langs_state = {}
        for lang in self.langs:
            # anchor = earliest buffered document with unconsumed tokens
            anchor = None
            for s in self.streams[lang].values():
                for seg in s.segments:
                    if seg.off < len(seg.tokens) and (anchor is None or seg.pos < anchor[0]):
                        anchor = (seg.pos, seg.seed_pred)
            head = self.readers[lang].head()
            langs_state[lang] = {
                "head": list(head),
                "anchor_pos": list(anchor[0]) if anchor else list(head),
                "anchor_seed": anchor[1] if anchor else self.composition_seeds[lang],
                "streams": {
                    c: {
                        "boundary_pos": list(s.boundary_pos) if s.boundary_pos else None,
                        "boundary_off": s.boundary_off,
                    }
                    for c, s in self.streams[lang].items()
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
            reader = self.readers[lang]
            head = tuple(st["head"])
            anchor_pos = tuple(st["anchor_pos"])

            assert set(st["streams"]) == set(self.streams[lang]), (
                f"resume composition set mismatch for '{lang}' — the tokenizer must be "
                "rebuilt identically (same seed) before restoring"
            )
            # restore consumption boundaries so replay drops what was consumed
            for c, b in st["streams"].items():
                stream = self.streams[lang][c]
                stream.boundary_pos = tuple(b["boundary_pos"]) if b["boundary_pos"] else None
                stream.boundary_off = b["boundary_off"]

            # replay from the anchor: re-roll the chain, re-tokenize, refill
            reader.seek(anchor_pos)
            self.composition_seeds[lang] = st["anchor_seed"]
            while reader.head() < head:
                self._route_next_doc(lang)
            assert reader.head() == head, (
                f"replay overshoot for '{lang}': {reader.head()} != {head} "
                "(corpus changed since the checkpoint?)"
            )
