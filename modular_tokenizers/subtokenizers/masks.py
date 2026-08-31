"""Language masks: which token ids of the global vocabulary belong to each
language. A mask is a boolean numpy array over the global id space; special
and byte tokens are always kept.

Extraction strategies (paper §3), i.e. where each language's vocabulary V_i
comes from:

  - ``merged_seq_bpe`` (sequential BPE, §3.2): the V_i were CREATED during
    sequential training and recorded in the tokenizer directory's ``lang_to_ids.json``;
    the mask just reads those ids.

  - ``merged_norm`` (merged Unigram, §3.1 — the paper's default Unigram
    setting): the global tokenizer is the union of monolingual vocabularies
    with probabilities RE-ESTIMATED by EM over the combined corpus
    ("normalized", see ``training/unigram/em_reestimate.py``). V_i = each
    language's original monolingual vocabulary, read from ``lang_to_ids.json``.

  - ``merged_unnorm``: same vocabulary union but WITHOUT EM re-estimation —
    each piece keeps the score from its own monolingual model. No directory:
    the monolingual .model files are passed directly (``path=en.model,fr.model``)
    and merged at load time; V_i = each model's own piece set.

  - ``common`` (frequency-based extraction, §3.1's alternative): works on ANY
    global tokenizer — V_i = its ``size`` most frequent tokens on language i's
    corpus (counts from ``training/unigram/token_counts.py``).

  - ``strategy=None``: no restriction — every language uses the full
    vocabulary (the baseline model's setting).
"""
import json
from collections import Counter
from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict, field_serializer, model_validator


class ExtractionMethod(Enum):
    """Where each language's vocabulary V_i comes from — see module docstring."""

    COMMON = "common"                  # top-k most frequent tokens per language
    MERGED_NORM = "merged_norm"        # union + EM re-estimation (Unigram default)
    MERGED_UNNORM = "merged_unnorm"    # union, original monolingual scores
    MERGED_SEQ_BPE = "merged_seq_bpe"  # V_i created by sequential BPE training


class ExtractionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: ExtractionMethod | None = None
    size: int | None = None                    # COMMON: tokens per language
    path_tokens_occurence: str | None = None   # COMMON: counts file
    path_lang_to_index: str | None = None      # MERGED_NORM / MERGED_SEQ_BPE
    # (may be filled in later from a tokenizer directory, so the merged
    #  strategies are validated at use time, in build_language_masks)

    @model_validator(mode="after")
    def validate_paths(self):
        if self.strategy == ExtractionMethod.COMMON:
            if not self.path_tokens_occurence:
                raise ValueError("COMMON strategy requires path_tokens_occurence")
            if not self.size:
                raise ValueError("COMMON strategy requires size")
        return self

    @field_serializer('strategy')
    def serialize_strategy(self, strategy: ExtractionMethod | None):
        return strategy.value if strategy else None


def build_special_tokens_mask(tokenizer, padded_size: int) -> np.ndarray:
    """Mask of the always-kept tokens (specials + byte pieces) of a
    ``Tokenizer`` wrapper, padded with False up to ``padded_size``."""
    mask = np.array(
        [tokenizer.is_special_token(i) for i in range(tokenizer.vocab_size())],
        dtype=bool,
    )
    pad = padded_size - tokenizer.vocab_size()
    return np.concatenate((mask, np.zeros(pad, dtype=bool)))


def build_language_masks(
    config: ExtractionConfig,
    langs: list[str],
    special_tokens_mask: np.ndarray,
    actual_vocab_size: int,
    *,
    monolingual_vocab_by_lang: dict | None = None,
    piece_to_id_global: dict | None = None,
) -> dict:
    """{lang: bool mask over the global id space}, per the chosen strategy.

    ``monolingual_vocab_by_lang`` ({lang: set of pieces of that language's
    monolingual tokenizer}) and ``piece_to_id_global`` are required for
    MERGED_UNNORM only.
    """
    masks = {}

    if config.strategy is None:
        for lang in langs:
            mask = np.zeros(len(special_tokens_mask), dtype=bool)
            mask[:actual_vocab_size] = True
            masks[lang] = mask

    elif config.strategy == ExtractionMethod.COMMON:
        with open(config.path_tokens_occurence, "r") as f:
            token_counts = json.load(f)

        num_special = int(special_tokens_mask.sum())
        max_content_tokens = config.size - num_special

        for lang in langs:
            candidates = [
                int(tid) for tid, _ in Counter(token_counts[lang]).most_common()
                if not special_tokens_mask[int(tid)]
            ]

            mask = special_tokens_mask.copy()
            mask[candidates[:max_content_tokens]] = True
            masks[lang] = mask

    elif config.strategy in (ExtractionMethod.MERGED_NORM, ExtractionMethod.MERGED_SEQ_BPE):
        if not config.path_lang_to_index:
            raise ValueError(
                f"{config.strategy.value} requires path_lang_to_index — set it, or "
                "point tokenizer.path at a tokenizer directory containing lang_to_ids.json"
            )
        with open(config.path_lang_to_index, "r") as f:
            lang_to_ids = json.load(f)

        for lang in langs:
            mask = special_tokens_mask.copy()
            mask[lang_to_ids[lang]] = True
            masks[lang] = mask

    elif config.strategy == ExtractionMethod.MERGED_UNNORM:
        assert monolingual_vocab_by_lang is not None and piece_to_id_global is not None

        for lang in langs:
            ids = [
                piece_to_id_global[piece]
                for piece in monolingual_vocab_by_lang.get(lang, [])
                if piece in piece_to_id_global
            ]
            mask = special_tokens_mask.copy()
            mask[ids] = True
            masks[lang] = mask

    return masks
