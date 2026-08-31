"""Unigram tokenizer training on one or more document sources.

The monolingual tokenizers of the paper are the single-source case; the
multilingual baseline is the multi-source case (all languages' corpora).
A source is any user path — a plain text file, a .zst file, a JSONL file (use
``text_field``), or a directory of such files; several sources can be
interleaved with optional weights instead of requiring a pre-concatenated
corpus. No particular directory organization is assumed.

CLI (key=value; a YAML can be passed as config=file.yaml):
    python -m modular_tokenizers.training.unigram.train \
        inputs=/any/path/to/fr_corpus.txt model_prefix=out/fr_24k vocab_size=24000
    python -m modular_tokenizers.training.unigram.train \
        inputs=/data/en.jsonl,/data/fr_shards_dir/ text_field=text \
        weights=0.7,0.3 model_prefix=out/en_fr vocab_size=48000
"""
import logging
import os

import sentencepiece as spm
from pydantic import BaseModel

from modular_tokenizers.training.utils import interleave_docs

logger = logging.getLogger(__name__)


def train_hf_unigram(
    sentences,
    output_path: str,
    vocab_size: int,
):
    """Train a Unigram tokenizer with HF `tokenizers` (alternative backend).

    Mirrors the SentencePiece settings: no normalization, whitespace
    pre-tokenization (Metaspace), digits split, byte fallback. HF's trainer
    does not emit byte pieces itself, so the 256 ``<0xXX>`` pieces are
    injected after training and the model's ``byte_fallback`` flag enabled.
    """
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers import decoders, pre_tokenizers
    from tokenizers.models import Unigram
    from tokenizers.trainers import UnigramTrainer

    from modular_tokenizers.training.utils import (
        BYTE_PIECES,
        SPECIAL_TOKENS,
        get_hf_json,
        hf_from_json,
    )

    tokenizer = HFTokenizer(Unigram())
    tokenizer.normalizer = None
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Metaspace(),
        pre_tokenizers.Digits(individual_digits=True),
    ])
    tokenizer.decoder = decoders.Sequence([
        decoders.Metaspace(),
        decoders.ByteFallback(),
        decoders.Fuse(),
    ])

    trainer = UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        unk_token="<unk>",
    )
    tokenizer.train_from_iterator(sentences, trainer)

    # SentencePiece's trainer creates the 256 byte pieces itself; HF's does
    # not, and byte_fallback only works if they exist in the vocab — so inject
    # them. HF never uses byte pieces in ordinary matching (they serve only
    # the fallback path, which ignores scores), so their score is inert; a low
    # value is kept so downstream tooling (e.g. EM re-scoring) treats them as
    # negligible-probability pieces.
    BYTE_FALLBACK_SCORE = -20.0
    cfg = get_hf_json(tokenizer)
    existing = {p for p, _ in cfg["model"]["vocab"]}
    for piece in BYTE_PIECES:
        if piece not in existing:
            cfg["model"]["vocab"].append([piece, BYTE_FALLBACK_SCORE])
    cfg["model"]["byte_fallback"] = True
    tokenizer = hf_from_json(cfg)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tokenizer.save(output_path)
    logger.info(f"Saved {output_path} (|vocab|={tokenizer.get_vocab_size()})")


def train(
    inputs: str,
    model_prefix: str,
    vocab_size: int,
    model_type: str = "unigram",
    backend: str = "hf",
    weights: str | None = None,
    text_field: str | None = None,
    input_sentence_size: int = 10_000_000,
    seed: int = 1234,
):
    """Train a tokenizer (paper settings: no normalization, byte fallback,
    digits split). ``backend='hf'`` (default) writes ``<model_prefix>.json``
    (HF `tokenizers` format, Unigram only); ``backend='sentencepiece'`` writes
    ``<model_prefix>.model`` and is the backend the paper's tokenizers were
    trained with (the two trainers are not algorithm-identical)."""
    from modular_tokenizers.config import as_str_list

    input_list = as_str_list(inputs)
    weight_list = [float(x) for x in as_str_list(weights)] if weights else None
    if weight_list is not None:
        assert len(weight_list) == len(input_list), "weights must match inputs"

    out_dir = os.path.dirname(model_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if backend == "hf":
        assert model_type == "unigram", "the HF backend supports unigram only"
        train_hf_unigram(
            interleave_docs(input_list, weight_list, text_field, seed),
            f"{model_prefix}.json",
            vocab_size,
        )
        return
    assert backend == "sentencepiece", f"unknown backend: {backend}"

    logger.info(f"Training SentencePiece {model_type} tokenizer: vocab={vocab_size}, sources={input_list}")

    spm.SentencePieceTrainer.train(
        sentence_iterator=interleave_docs(input_list, weight_list, text_field, seed),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        input_sentence_size=input_sentence_size,
        num_threads=os.cpu_count(),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        pad_id=3,
        train_extremely_large_corpus=True,
    )

    logger.info(f"Saved {model_prefix}.model")


class TrainArgs(BaseModel):
    inputs: str | None = None
    model_prefix: str | None = None
    vocab_size: int | None = None
    model_type: str = "unigram"
    backend: str = "hf"
    weights: str | None = None
    text_field: str | None = None
    input_sentence_size: int = 10_000_000
    seed: int = 1234


def main():
    from modular_tokenizers.config import parse_args_to_pydantic_model, setup_logging

    setup_logging(os.environ.get("LOGLEVEL"))
    args = parse_args_to_pydantic_model(TrainArgs)
    assert args.inputs and args.model_prefix and args.vocab_size, \
        "inputs, model_prefix and vocab_size are required"
    train(**args.model_dump())


if __name__ == "__main__":
    main()
