"""The modular tokenizer: one global vocabulary, served per subtokenizer.

Paper picture (§3): a *global* tokenizer T holds the full multilingual
vocabulary; a *subtokenizer* T_s restricts it to a subset of languages and is
identified by them — ``"fr"`` (monolingual), ``"en,fr"`` (unified), ``"all"``
(= T itself).

Division of labor in this package:

    tokenizer.py    the plain SP/HF wrapper (what a "tokenizer" is)
    masks.py        which global token ids belong to each language
    extraction.py   turning a mask into an actual subtokenizer
    sampling.py     which subtokenizer each training batch samples
    THIS FILE       gluing it together: load T, cache subtokenizers,
                    encode/decode per subtokenizer

Three tokenizer kinds are supported, detected from the loaded file itself:
``tokenizer.model`` -> SentencePiece Unigram; ``tokenizer.json`` -> HF, with
its ``model.type`` field deciding Unigram vs BPE. ``path`` may be the model
file, a training-output directory (the model file is found inside, and a
``lang_to_ids.json`` next to it becomes the default mask mapping), or a
comma-separated list of monolingual SentencePiece models (merged at load
time — MERGED_UNNORM).

Token ids seen by callers are ALWAYS in the global id space. Internally, a
subtokenizer holds only its own tokens with contiguous ids, and
``encode``/``decode`` translate through sub<->global id maps.
"""
import dataclasses
import json
import os

import numpy as np
from pydantic import BaseModel, ConfigDict
from sentencepiece import SentencePieceProcessor

from modular_tokenizers.subtokenizers.sampling import SamplingConfig
from modular_tokenizers.subtokenizers.extraction import (
    bpe_pieces_by_lang,
    build_id_maps,
    get_vocab_and_merges,
    materialize_subtokenizer,
)
from modular_tokenizers.subtokenizers.masks import (
    ExtractionConfig,
    ExtractionMethod,
    build_language_masks,
    build_special_tokens_mask,
)
from modular_tokenizers.subtokenizers.tokenizer import Tokenizer

# File basenames inside a tokenizer directory (see training/*/serialization.py).
MODEL_BASENAMES = ("tokenizer.model", "tokenizer.json")
LANG_TO_IDS_BASENAME = "lang_to_ids.json"


# ---------------------------------------------------------------------------
# Loading the global tokenizer
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class GlobalTokenizer:
    """The loaded global tokenizer T, with what was detected from its file."""

    tokenizer: Tokenizer
    algorithm: str                    # "unigram" | "bpe"
    lang_to_ids_path: str | None = None    # lang_to_ids.json found next to the model, if any
    token_to_merge_rule: dict | None = None  # BPE only: token -> the (a, b) rule that forms it
    monolingual_vocab_by_lang: dict | None = None  # MERGED_UNNORM only: {lang: set of pieces}


def load_global_tokenizer(path: str, langs: str, extraction: ExtractionConfig) -> GlobalTokenizer:
    """Load T from a model file, a training-output directory, or a
    comma-separated list of monolingual SentencePiece models (merged at load
    time). Detects the algorithm (and backend) from the file itself."""
    paths = path.split(',')

    if len(paths) > 1:
        return _load_and_merge_monolingual(paths, langs.split(','), extraction)

    model_file, lang_to_ids_path = _find_model_file(paths[0])
    tokenizer = Tokenizer(path=model_file)

    if not tokenizer.use_hf:
        return GlobalTokenizer(tokenizer, "unigram", lang_to_ids_path)

    model_type = json.loads(tokenizer.tokenizer.to_str())["model"]["type"]
    if model_type == "Unigram":
        return GlobalTokenizer(tokenizer, "unigram", lang_to_ids_path)
    if model_type == "BPE":
        assert extraction.strategy in (None, ExtractionMethod.MERGED_SEQ_BPE), \
            "a sequential-BPE tokenizer uses the merged_seq_bpe extraction strategy"
        _, merge_rules = get_vocab_and_merges(tokenizer.tokenizer)
        token_to_merge_rule = {"".join(rule): rule for rule in merge_rules}
        return GlobalTokenizer(tokenizer, "bpe", lang_to_ids_path, token_to_merge_rule)
    raise ValueError(f"Unsupported HF model type: {model_type}")


def _find_model_file(path: str) -> tuple[str, str | None]:
    """Resolve ``path`` to (model_file, lang_to_ids.json path or None).

    A directory (a training-output folder) is searched for the known model
    basenames, and for the ``lang_to_ids.json`` written next to them."""
    if not os.path.isdir(path):
        sibling = os.path.join(os.path.dirname(path), LANG_TO_IDS_BASENAME)
        return path, sibling if os.path.isfile(sibling) else None

    for name in MODEL_BASENAMES:
        model_file = os.path.join(path, name)
        if os.path.isfile(model_file):
            lang_to_ids = os.path.join(path, LANG_TO_IDS_BASENAME)
            return model_file, lang_to_ids if os.path.isfile(lang_to_ids) else None
    raise FileNotFoundError(f"no {' or '.join(MODEL_BASENAMES)} found in directory {path}")


def _load_and_merge_monolingual(paths, langs, extraction) -> GlobalTokenizer:
    """MERGED_UNNORM: T is built at load time as the vocabulary union of
    monolingual SentencePiece models (no probability re-estimation)."""
    assert extraction.strategy == ExtractionMethod.MERGED_UNNORM
    assert len(paths) == len(langs), "Path count must match Lang count"
    assert not any(p.endswith(".json") for p in paths), \
        "merging at load time (MERGED_UNNORM) is SentencePiece-only"

    from modular_tokenizers.training.unigram.merge import merge_tokenizers

    monolingual = {lang: SentencePieceProcessor(model_file=p) for lang, p in zip(langs, paths)}
    merged = merge_tokenizers([monolingual[l] for l in sorted(langs)])

    monolingual_vocab_by_lang = {
        lang: {proc.IdToPiece(i) for i in range(proc.vocab_size())}
        for lang, proc in monolingual.items()
    }

    return GlobalTokenizer(
        Tokenizer(spm_tokenizer=merged), "unigram",
        monolingual_vocab_by_lang=monolingual_vocab_by_lang,
    )


# ---------------------------------------------------------------------------
# The modular tokenizer
# ---------------------------------------------------------------------------

class ModularTokenizer:
    """Serve the subtokenizers of a global tokenizer.

    ``encode``/``decode`` work like ``Tokenizer``'s, with one extra argument:
    the id of the subtokenizer (= its languages) whose vocabulary to use.
    Subtokenizers are materialized on first use and cached.
    """

    def __init__(
        self,
        *,
        path: str | None = None,
        langs: str | None = None,
        pad_vocab: int | None = None,
        sampling_strategy: SamplingConfig | None = None,
        extraction_strategy: ExtractionConfig | None = None,
    ):
        self.langs = sorted(langs.split(','))
        self.pad_vocab = pad_vocab
        self._cached_vocab_size = None
        self.sampling_strategy = sampling_strategy or SamplingConfig()
        # Own copy: defaults resolved below must not leak into the caller's config.
        self.extraction_strategy = (extraction_strategy or ExtractionConfig()).model_copy(deep=True)

        # The global tokenizer T; backend/algorithm are detected from its file.
        global_tokenizer = load_global_tokenizer(path, langs, self.extraction_strategy)
        self.tokenizer = global_tokenizer.tokenizer
        self.algorithm = global_tokenizer.algorithm
        self._token_to_merge_rule = global_tokenizer.token_to_merge_rule

        # A lang_to_ids.json found next to the model file is the default mapping.
        if self.extraction_strategy.path_lang_to_index is None:
            self.extraction_strategy.path_lang_to_index = global_tokenizer.lang_to_ids_path

        # Which global token ids belong to each language.
        self.special_tokens_mask = build_special_tokens_mask(self.tokenizer, self.vocab_size())
        self.masks = build_language_masks(
            self.extraction_strategy,
            self.langs,
            self.special_tokens_mask,
            self.tokenizer.vocab_size(),
            monolingual_vocab_by_lang=global_tokenizer.monolingual_vocab_by_lang,
            piece_to_id_global=self.tokenizer.pieces_to_id_mapping,
        )
        self._bpe_pieces_by_lang = (
            bpe_pieces_by_lang(self.tokenizer, self.masks, self.langs)
            if self.algorithm == "bpe" else None
        )

        # Which subtokenizers each language's batches may sample at training
        # time (Full / Data(n)), with their probabilities.
        self.subtokenizer_ids_by_lang, self.subtokenizer_probs = \
            self.sampling_strategy.build_subtokenizer_ids_by_lang(self.langs)

        self._prebuild_subtokenizers()

    def _prebuild_subtokenizers(self):
        """Warm the cache with every subtokenizer training uses: each language,
        'all', and the sampled unified ids (as subtokenizers_by_lang)."""
        self.subtokenizers = {}
        self._sub_to_global_ids = {}
        self._global_to_sub_ids = {}

        for subtokenizer_id in self.langs + ['all']:
            self.build_subtokenizer(subtokenizer_id)

        self.subtokenizers_by_lang = {
            lang: {i: self.build_subtokenizer(i) for i in ids}
            for lang, ids in self.subtokenizer_ids_by_lang.items()
        }

    def encode(self, text, subtokenizer_id, bos=True, eos=True, return_mask=False):
        """Tokenize with the given subtokenizer (built on demand).
        Returned ids are in the global id space."""
        subtokenizer = self.build_subtokenizer(subtokenizer_id)
        tokens = subtokenizer.encode(text, bos=bos, eos=eos)
        tokens = self._sub_to_global_ids[subtokenizer_id][np.asarray(tokens, dtype=np.int32)]

        if return_mask:
            return tokens, self.masks[subtokenizer_id]
        return tokens

    def decode(self, tokens, subtokenizer_id, strip_bos=False, strip_eos=False):
        subtokenizer = self.build_subtokenizer(subtokenizer_id)
        tokens = self._global_to_sub_ids[subtokenizer_id][np.asarray(tokens, dtype=np.int32)]

        return subtokenizer.decode(tokens, strip_bos=strip_bos, strip_eos=strip_eos)

    def build_subtokenizer(self, subtokenizer_id: str) -> Tokenizer:
        """Return the subtokenizer for an id like 'en,fr', materializing and
        caching it (and its sub<->global id maps) on first use."""
        if subtokenizer_id in self.subtokenizers:
            return self.subtokenizers[subtokenizer_id]

        self.mask(subtokenizer_id)  # ensure the subtokenizer's mask is cached

        if subtokenizer_id == 'all':
            subtokenizer = self.tokenizer  # 'all' IS the global tokenizer
            identity = np.arange(self.tokenizer.vocab_size())
            id_maps = (identity, identity)
        else:
            subtokenizer = materialize_subtokenizer(
                self.tokenizer,
                self.algorithm,
                self.masks[subtokenizer_id],
                subtokenizer_id=subtokenizer_id,
                bpe_pieces_by_lang=self._bpe_pieces_by_lang,
                tokens_to_merges=self._token_to_merge_rule,
            )
            id_maps = build_id_maps(subtokenizer, self.tokenizer)

        self.subtokenizers[subtokenizer_id] = subtokenizer
        self._sub_to_global_ids[subtokenizer_id], self._global_to_sub_ids[subtokenizer_id] = id_maps
        return subtokenizer

    def mask(self, subtokenizer_id: str) -> np.ndarray:
        """Boolean mask over the global id space: which tokens the subtokenizer
        may use — the union of its languages' masks (+ special tokens)."""
        if subtokenizer_id not in self.masks:
            if subtokenizer_id == 'all':
                mask = np.zeros(self.vocab_size(), dtype=bool)
                mask[:self.tokenizer.vocab_size()] = True
            else:
                lang_masks = [self.masks[l] for l in subtokenizer_id.split(',') if l in self.masks]
                if lang_masks:
                    mask = np.logical_or.reduce(lang_masks) | self.special_tokens_mask
                else:
                    mask = self.special_tokens_mask.copy()
            self.masks[subtokenizer_id] = mask
        return self.masks[subtokenizer_id]

    # Ids and sizes: same interface as Tokenizer, delegated to the global T.

    def vocab_size(self):
        """T's vocab size, optionally rounded up to a multiple of ``pad_vocab``
        (models like their output dimension divisible by the sharding)."""
        if self._cached_vocab_size is None:
            size = self.tokenizer.vocab_size()
            if self.pad_vocab:
                size = ((size + self.pad_vocab - 1) // self.pad_vocab) * self.pad_vocab
            self._cached_vocab_size = size
        return self._cached_vocab_size

    def bos_id(self):
        return self.tokenizer.bos_id()

    def eos_id(self):
        return self.tokenizer.eos_id()

    def pad_id(self):
        return self.tokenizer.pad_id()

    def unk_id(self):
        return self.tokenizer.unk_id()


# ---------------------------------------------------------------------------
# Config entry point
# ---------------------------------------------------------------------------

class TokenizerArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str = "base"                    # "base" | "modular"
    name: str | None = None
    alpha: float | None = None
    path: str | None = None
    langs: str | None = None
    pad_vocab: int | None = None
    sampling_strategy: SamplingConfig | None = None
    extraction_strategy: ExtractionConfig | None = None

    def build_tokenizer(self):
        if self.type == "base":
            return Tokenizer(path=self.path)

        if self.type == "modular":
            return ModularTokenizer(
                path=self.path,
                langs=self.langs,
                pad_vocab=self.pad_vocab,
                sampling_strategy=self.sampling_strategy,
                extraction_strategy=self.extraction_strategy,
            )

        raise ValueError(f"Unknown tokenizer type: {self.type}")
