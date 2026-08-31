"""Subtokenizer extraction: turn a language mask into an actual tokenizer.

``materialize_subtokenizer`` is the entry point; it dispatches on the global
tokenizer's kind to one of three mechanisms:

  - SentencePiece Unigram: proto surgery (keep only the mask's pieces);
  - HF Unigram: filter the tokenizer.json vocab list;
  - HF BPE: rebuild vocab + merge table, keeping merges in global order —
    which is what guarantees every token stays constructible (the paper's
    properness property).

Terminology: "merges" here always means BPE merge *rules* (the ordered
(a, b) -> ab table) — not the merging of tokenizers, which is construction
(``training/unigram/merge.py``), nor the ``merged_*`` mask strategies
(``subtokenizers/masks.py``).
"""
import json
from typing import Dict, List, Tuple

import numpy as np
import sentencepiece
from sentencepiece import sentencepiece_model_pb2
from tokenizers import Tokenizer as HFTokenizer

from modular_tokenizers.subtokenizers.tokenizer import Tokenizer


def materialize_subtokenizer(
    global_tokenizer: Tokenizer,
    algorithm: str,
    mask: np.ndarray,
    *,
    subtokenizer_id: str | None = None,
    bpe_pieces_by_lang: dict | None = None,
    tokens_to_merges: dict | None = None,
) -> Tokenizer:
    """Build one subtokenizer of a global tokenizer.

    The kept tokens get contiguous new ids; ``build_id_maps`` provides the
    translation to/from global ids.
    """
    if algorithm == "bpe":
        raw = extract_hf_bpe_subtokenizer(
            global_tokenizer.tokenizer, subtokenizer_id, bpe_pieces_by_lang, tokens_to_merges
        )
        return Tokenizer(hf_tokenizer=raw)

    if global_tokenizer.use_hf:
        raw = extract_hf_unigram_subtokenizer(global_tokenizer.tokenizer, mask)
        return Tokenizer(hf_tokenizer=raw)

    raw = extract_unigram_subtokenizer(global_tokenizer.tokenizer, mask)
    return Tokenizer(spm_tokenizer=raw)


# ---------------------------------------------------------------------------
# Unigram — SentencePiece (proto surgery)
# ---------------------------------------------------------------------------

def extract_unigram_subtokenizer(tokenizer, tokens_to_keep):
    """Restrict a SentencePiece Unigram model to a token mask (kept pieces are
    re-indexed contiguously)."""
    m = sentencepiece_model_pb2.ModelProto()
    m.ParseFromString(tokenizer.serialized_model_proto())

    assert m.trainer_spec.model_type == 1
    m_new = sentencepiece_model_pb2.ModelProto()
    m_new.trainer_spec.CopyFrom(m.trainer_spec)
    m_new.normalizer_spec.CopyFrom(m.normalizer_spec)

    for idx, piece in enumerate(m.pieces):
        if tokens_to_keep[idx]:
            m_new.pieces.append(piece)

    tokenizer_new = sentencepiece.SentencePieceProcessor()
    tokenizer_new.LoadFromSerializedProto(m_new.SerializeToString())

    return tokenizer_new


# ---------------------------------------------------------------------------
# Unigram — HF tokenizers (vocab-list surgery)
# ---------------------------------------------------------------------------

def extract_hf_unigram_subtokenizer(tokenizer, tokens_to_keep):
    """Restrict an HF `tokenizers` Unigram model to a token mask (the HF
    counterpart of ``extract_unigram_subtokenizer``)."""
    raw = getattr(tokenizer, "_tokenizer", tokenizer)
    cfg = json.loads(raw.to_str())
    assert cfg["model"]["type"] == "Unigram"

    vocab = cfg["model"]["vocab"]  # list of [piece, log_prob]
    unk_piece = vocab[cfg["model"]["unk_id"]][0] if cfg["model"]["unk_id"] is not None else None

    new_vocab = [[piece, score] for idx, (piece, score) in enumerate(vocab) if tokens_to_keep[idx]]

    cfg["model"]["vocab"] = new_vocab
    if unk_piece is not None:
        cfg["model"]["unk_id"] = next(i for i, (p, _) in enumerate(new_vocab) if p == unk_piece)

    return HFTokenizer.from_str(json.dumps(cfg))


# ---------------------------------------------------------------------------
# BPE — HF tokenizers (vocab + merge-table surgery)
# ---------------------------------------------------------------------------

def get_vocab_and_merges(tokenizer) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    """Vocab and merge rules of an HF BPE tokenizer.

    Accepts either a raw ``tokenizers.Tokenizer`` or a
    ``transformers.PreTrainedTokenizerFast`` wrapper.
    """
    raw = getattr(tokenizer, "_tokenizer", tokenizer)
    cfg = json.loads(raw.to_str())
    merges = [tuple(x) if isinstance(x, list) else tuple(x.split(" ")) for x in cfg["model"]["merges"]]
    vocab = cfg["model"]["vocab"]
    return vocab, merges


def replace_vocab_and_merges(tokenizer, vocab, merges):
    """Return a tokenizer whose BPE vocab/merge rules are replaced.

    A raw ``tokenizers.Tokenizer`` input yields a new raw tokenizer; a
    ``PreTrainedTokenizerFast`` input is updated in place (and returned).
    """
    raw = getattr(tokenizer, "_tokenizer", tokenizer)
    tokenizer_json = json.loads(raw.to_str())
    tokenizer_json["model"]["merges"] = [" ".join(m) for m in merges]
    tokenizer_json["model"]["vocab"] = vocab
    new_raw = HFTokenizer.from_str(json.dumps(tokenizer_json))

    if hasattr(tokenizer, "_tokenizer"):
        tokenizer._tokenizer = new_raw
        return tokenizer
    return new_raw


def extract_hf_bpe_subtokenizer(tokenizer, subtokenizer_id, pieces_and_index_by_lang, tokens_to_merges):
    """Build a subtokenizer from a global BPE
    tokenizer: union of the languages' vocabularies, re-indexed contiguously,
    with each token's merge rule kept in global order."""
    vocab = {}
    for l in subtokenizer_id.split(','):
        for token, idx in pieces_and_index_by_lang[l].items():
            if token not in vocab:
                vocab[token] = idx

    vocab = sorted(list(vocab.items()), key=lambda x: x[1])
    new_vocab, new_merges = {}, []

    for token, idx in vocab:
        new_vocab[token] = len(new_vocab)
        if token in tokens_to_merges:
            new_merges.append(tokens_to_merges[token])

    return replace_vocab_and_merges(tokenizer, new_vocab, new_merges)


def bpe_pieces_by_lang(global_tokenizer: Tokenizer, language_masks: dict, langs: list[str]) -> dict:
    """{lang: {piece: global id}} for each language — the lookup
    ``extract_hf_bpe_subtokenizer`` needs to union language vocabularies."""
    return {
        lang: {
            global_tokenizer.id_to_pieces_mapping[i]: i
            for i in np.flatnonzero(language_masks[lang])
        }
        for lang in langs
    }


# ---------------------------------------------------------------------------
# Sub <-> global id maps
# ---------------------------------------------------------------------------

def mapping_dict_to_np(mapping):
    size = max(mapping.keys()) + 1
    arr = np.full(size, -1, dtype=np.int32)
    for k, v in mapping.items():
        arr[k] = v
    return arr


def build_id_maps(subtokenizer: Tokenizer, global_tokenizer: Tokenizer):
    """(sub -> global, global -> sub) id arrays for one subtokenizer, matching
    the subtokenizer's contiguous ids to global ids by piece string."""
    sub_to_global, global_to_sub = {}, {}

    for piece, sub_id in subtokenizer.pieces_to_id_mapping.items():
        global_id = global_tokenizer.pieces_to_id_mapping[piece]
        sub_to_global[sub_id] = global_id
        global_to_sub[global_id] = sub_id

    return mapping_dict_to_np(sub_to_global), mapping_dict_to_np(global_to_sub)
