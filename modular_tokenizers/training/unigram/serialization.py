"""Serialization of the merged-Unigram modular tokenizer: the on-disk layout,
its writer, loader, and validator.

``training/unigram/merge.py`` writes it (and ``em_reestimate.py`` replaces its
model with the EM-rescored one); subtokenizer extraction and model training
read it. Layout:

    <tokenizer_dir>/
        tokenizer.model  OR  tokenizer.json   # merged Unigram model
                                              # (.model: SentencePiece, the paper backend;
                                              #  .json: HF `tokenizers` backend)
        lang_to_ids.json                      # {lang: [ids of that language's pieces
                                              #         in the merged vocab]}

``lang_to_ids.json`` follows the same schema as the sequential-BPE one
(``training/bpe/serialization.py``), so the consumption side
(``ExtractionConfig(strategy="merged_norm", path_lang_to_index=...)``) is
identical for both methods and both backends.
"""
import json
import logging
import os

logger = logging.getLogger(__name__)

SP_MODEL_NAME = "tokenizer.model"
HF_MODEL_NAME = "tokenizer.json"


def model_path(tokenizer_dir: str) -> str:
    """Path of the directory's model file, whichever backend produced it."""
    for name in (SP_MODEL_NAME, HF_MODEL_NAME):
        path = os.path.join(tokenizer_dir, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"no {SP_MODEL_NAME} or {HF_MODEL_NAME} in {tokenizer_dir}")


def lang_to_ids_path(tokenizer_dir: str) -> str:
    return os.path.join(tokenizer_dir, "lang_to_ids.json")


def write(
    out_dir: str,
    *,
    lang_to_ids: dict,
    merged_model_proto: bytes | None = None,
    merged_hf_tokenizer=None,
) -> None:
    """Write the tokenizer directory. Pass exactly one of ``merged_model_proto``
    (SentencePiece serialized proto) or ``merged_hf_tokenizer``
    (HF ``tokenizers.Tokenizer``)."""
    assert (merged_model_proto is None) != (merged_hf_tokenizer is None), \
        "pass exactly one of merged_model_proto / merged_hf_tokenizer"
    os.makedirs(out_dir, exist_ok=True)

    if merged_model_proto is not None:
        with open(os.path.join(out_dir, SP_MODEL_NAME), "wb") as f:
            f.write(merged_model_proto)
    else:
        merged_hf_tokenizer.save(os.path.join(out_dir, HF_MODEL_NAME))

    with open(lang_to_ids_path(out_dir), "w", encoding="utf-8") as f:
        json.dump(lang_to_ids, f, indent=2, ensure_ascii=False)

    logger.info(f"tokenizer directory written -> {out_dir}")


def load_lang_to_ids(tokenizer_dir: str) -> dict:
    with open(lang_to_ids_path(tokenizer_dir), "r", encoding="utf-8") as f:
        return json.load(f)


def _vocab_size(model_file: str) -> int:
    if model_file.endswith(".json"):
        from tokenizers import Tokenizer as HFTokenizer

        return HFTokenizer.from_file(model_file).get_vocab_size()
    from sentencepiece import SentencePieceProcessor

    return SentencePieceProcessor(model_file=model_file).vocab_size()


def validate(tokenizer_dir: str, langs: list[str] | None = None) -> None:
    """Cheap structural validation: files present, ids in range."""
    model_file = model_path(tokenizer_dir)
    if not os.path.isfile(lang_to_ids_path(tokenizer_dir)):
        raise FileNotFoundError(f"tokenizer directory is missing {lang_to_ids_path(tokenizer_dir)}")

    vocab_size = _vocab_size(model_file)

    lang_to_ids = load_lang_to_ids(tokenizer_dir)
    langs = langs or list(lang_to_ids.keys())

    for lang in langs:
        if lang not in lang_to_ids:
            raise KeyError(f"language '{lang}' not in lang_to_ids.json")
        ids = lang_to_ids[lang]
        if any(i < 0 or i >= vocab_size for i in ids):
            raise ValueError(f"lang_to_ids['{lang}'] contains ids outside the merged vocab")

    logger.info(f"tokenizer directory at {tokenizer_dir} validated for {len(langs)} languages")
