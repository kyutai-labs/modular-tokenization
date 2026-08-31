"""Serialization of the trained sequential-BPE modular tokenizer: the on-disk
layout, its writer, loader, and validator.

``training/bpe/sequential.py`` *writes* it; subtokenizer extraction and model
training *read* it. The Unigram method has its own serialization
(``training/unigram/serialization.py``); both share the ``lang_to_ids.json``
convention so consumers don't care which method built the folder. Layout:

    <tokenizer_dir>/
        tokenizer.json              # the global tokenizer (HF `tokenizers` format)
        tokenizer_config.json
        special_tokens_map.json
        vocab.json                  # global vocab {token: id}
        merges.json                 # global merges ["a b", ...]
        lang_to_ids.json            # {lang: [ids of that language's tokens in the GLOBAL vocab]}
        vocab_by_lang.json          # {lang: {token: id in that language's subtokenizer}}
        lang_to_merges.json         # {lang: [[a, b], ...]}
        extracted/<lang>/           # standalone per-language subtokenizer
            tokenizer.json  tokenizer_config.json  special_tokens_map.json
            vocab.json  merges.json

``lang_to_ids.json`` is the file consumed at model-training time
(``ExtractionConfig(strategy="merged_seq_bpe", path_lang_to_index=...)``); the
``extracted/`` subtokenizers support standalone monolingual use.
"""
import json
import logging
import os

from tokenizers import Tokenizer as HFTokenizer
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

MAPPING_FILES = ("lang_to_ids.json", "vocab_by_lang.json", "lang_to_merges.json")


def _dump(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_tokenizer_files(tokenizer: PreTrainedTokenizerFast, out_dir: str, vocab, merges) -> None:
    os.makedirs(out_dir, exist_ok=True)
    tokenizer.save_pretrained(out_dir)
    _dump(os.path.join(out_dir, "vocab.json"), vocab)
    _dump(os.path.join(out_dir, "merges.json"), [" ".join(m) for m in merges])


def write(
    out_dir: str,
    *,
    global_tokenizer: PreTrainedTokenizerFast,
    global_vocab: dict,
    global_merges: list,
    per_lang: dict,
    langs_order: list[str],
) -> None:
    """Write a complete tokenizer directory.

    ``per_lang`` maps each language to a ``(vocab, merges, tokenizer)`` triple
    for its extracted subtokenizer, where ``vocab`` is ``{token: id in the
    subtokenizer}`` and ``merges`` a list of ``(a, b)`` pairs.
    """
    os.makedirs(out_dir, exist_ok=True)

    vocab_by_lang, lang_to_merges, lang_to_ids = {}, {}, {}
    for lang in langs_order:
        vocab, merges, _ = per_lang[lang]
        vocab_by_lang[lang] = vocab
        lang_to_merges[lang] = merges
        lang_to_ids[lang] = [global_vocab[t] for t in vocab]
        logger.info(f"{lang}: |vocab|={len(vocab)} |merges|={len(merges)}")

    _dump(os.path.join(out_dir, "vocab_by_lang.json"), vocab_by_lang)
    _dump(os.path.join(out_dir, "lang_to_merges.json"), lang_to_merges)
    _dump(os.path.join(out_dir, "lang_to_ids.json"), lang_to_ids)

    _save_tokenizer_files(global_tokenizer, out_dir, global_vocab, global_merges)
    for lang in langs_order:
        vocab, merges, tokenizer = per_lang[lang]
        _save_tokenizer_files(tokenizer, os.path.join(out_dir, "extracted", lang), vocab, merges)

    logger.info(f"tokenizer directory written -> {out_dir} (global |vocab|={len(global_vocab)})")


def global_tokenizer_path(tokenizer_dir: str) -> str:
    return os.path.join(tokenizer_dir, "tokenizer.json")


def lang_to_ids_path(tokenizer_dir: str) -> str:
    return os.path.join(tokenizer_dir, "lang_to_ids.json")


def load_lang_to_ids(tokenizer_dir: str) -> dict:
    return _load_json(lang_to_ids_path(tokenizer_dir))


def load_global_tokenizer(tokenizer_dir: str) -> HFTokenizer:
    return HFTokenizer.from_file(global_tokenizer_path(tokenizer_dir))


def validate(tokenizer_dir: str, langs: list[str] | None = None) -> None:
    """Cheap structural validation: files present, ids in range, extracted
    vocabularies consistent with ``lang_to_ids``. Raises on the first problem."""
    for name in ("tokenizer.json",) + MAPPING_FILES:
        path = os.path.join(tokenizer_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"tokenizer directory is missing {name} ({path})")

    global_tok = load_global_tokenizer(tokenizer_dir)
    global_vocab = global_tok.get_vocab()
    vocab_size = len(global_vocab)

    lang_to_ids = load_lang_to_ids(tokenizer_dir)
    langs = langs or list(lang_to_ids.keys())

    for lang in langs:
        if lang not in lang_to_ids:
            raise KeyError(f"language '{lang}' not in lang_to_ids.json")
        ids = lang_to_ids[lang]
        if any(i < 0 or i >= vocab_size for i in ids):
            raise ValueError(f"lang_to_ids['{lang}'] contains ids outside the global vocab")

        extracted_path = os.path.join(tokenizer_dir, "extracted", lang, "tokenizer.json")
        if os.path.isfile(extracted_path):
            sub_vocab = HFTokenizer.from_file(extracted_path).get_vocab()
            missing = [t for t in sub_vocab if t not in global_vocab]
            if missing:
                raise ValueError(
                    f"extracted/{lang} has {len(missing)} tokens absent from the "
                    f"global vocab (e.g. {missing[:5]})"
                )
            if len(sub_vocab) != len(ids):
                raise ValueError(
                    f"extracted/{lang}: |vocab|={len(sub_vocab)} but "
                    f"lang_to_ids has {len(ids)} ids"
                )

    logger.info(f"tokenizer directory at {tokenizer_dir} validated for {len(langs)} languages")
