"""Sequential BPE training (paper §3.2, Algorithm 1).

Builds one multilingual tokenizer language by language: the first language is
trained from scratch; every subsequent language runs continued BPE training on
top of the current global tokenizer. At each language step, both the growing
global tokenizer and a fixed-budget per-language subtokenizer (closed under
its merges, so every token stays constructible) are produced. The run ends by
writing the tokenizer directory (see ``modular_tokenizers.training.bpe.serialization``).

CLI (key=value; a YAML can be passed as config=file.yaml):
    python -m modular_tokenizers.training.bpe.sequential \
        input_paths=bg.txt,cs.txt,... langs_order=bg,cs,... \
        vocab_size=24000 output_dir=out/
"""
import gc
import logging
import os
from copy import deepcopy
from typing import Optional

from pydantic import BaseModel

from modular_tokenizers.training.bpe import serialization
from modular_tokenizers.training.bpe.continued_training import train_vocab_extension
from modular_tokenizers.subtokenizers.extraction import (
    get_vocab_and_merges,
    replace_vocab_and_merges,
)
from modular_tokenizers.training.utils import iter_docs, write_json
from modular_tokenizers.training.bpe.train_base import train_bpe_from_scratch

logger = logging.getLogger(__name__)


def save_step(tokenizer, output_path, vocab, merges):
    """Save one step's tokenizer + raw vocab/merges (intermediate artifact,
    useful for inspection and for warm-restarting a long run)."""
    os.makedirs(output_path, exist_ok=True)

    tokenizer.save_pretrained(os.path.join(output_path, "tokenizer"))
    write_json(vocab, os.path.join(output_path, "vocab.json"))
    write_json([" ".join(x) for x in merges], os.path.join(output_path, "merges.json"))


def learn_extension(
        input_path: str,
        tokenizer,
        vocab_size: int,
        max_token_length: Optional[int] = None,
        max_sentences: int = -1,
        text_field: Optional[str] = None,
):
    """Run continued BPE training on one language's corpus; returns the
    candidate new (vocab, merges). ``input_path`` is any user path (plain
    text, .zst, JSONL via ``text_field``, or a directory of such files)."""
    train_docs = list(iter_docs(input_path, text_field, max_sentences))

    gc.collect()
    extension_tokens = train_vocab_extension(
        tokenizer=tokenizer,
        corpus=train_docs,
        extension_size=vocab_size,
        max_token_length=max_token_length,
    )

    return extension_tokens["vocab"], extension_tokens["merges"]


def recursive_add(token, vocab, vocab_base, token_to_merges, idx=None):
    """Add ``token`` to ``vocab`` together with every ancestor unit its merge
    chain needs (keeps the extracted subtokenizer closed under merges)."""
    if token in vocab:
        return

    if token not in token_to_merges:
        vocab[token] = vocab_base[token]
        return

    if token in vocab_base:
        vocab[token] = vocab_base[token]
    else:
        vocab[token] = idx

    a, b = token_to_merges[token]

    assert (a in vocab or a in vocab_base) and (b in vocab or b in vocab_base)

    recursive_add(a, vocab, vocab_base, token_to_merges)
    recursive_add(b, vocab, vocab_base, token_to_merges)


def extract_vocab(new_merges, vocab_size, vocab_base, token_to_merges):
    """Select the per-language subtokenizer vocabulary (budget ``vocab_size``)
    from the continued-training output, walking merges in order and pulling in
    ancestors as needed (paper Algorithm 1: unigram-vs-bigram bookkeeping)."""
    vocab_extracted = {elt: vocab_base[elt] for elt in vocab_base if elt not in token_to_merges}

    idx_ext = len(vocab_base)

    for merge in new_merges:
        token = "".join(merge)

        if len(merge) == 1:
            recursive_add(token, vocab_extracted, vocab_base, token_to_merges)
        else:
            recursive_add(token, vocab_extracted, vocab_base, token_to_merges, idx_ext)
            idx_ext += 1

        if len(vocab_extracted) >= vocab_size:
            break

    vocab_extracted = dict(sorted(vocab_extracted.items(), key=lambda x: x[1]))

    new_vocab_extracted = {}

    for i, token in enumerate(vocab_extracted.keys()):
        new_vocab_extracted[token] = i
        if len(new_vocab_extracted) == vocab_size:
            break

    return new_vocab_extracted


def compute_extension_and_extraction(vocab_base, merges_base, new_vocab, new_merges, vocab_size, tokenizer_base):
    """From one language's continued-training output, produce
    (a) the fixed-budget extracted subtokenizer for that language and
    (b) the extended global tokenizer."""
    token_to_merges = {"".join(elt): elt for elt in merges_base}

    for merge in new_merges:
        for k in merge:
            assert (k in token_to_merges) or (k in vocab_base)

        if len(merge) == 2:
            token_to_merges["".join(merge)] = merge

    vocab_extracted = extract_vocab(new_merges, vocab_size, vocab_base, token_to_merges)

    merges_extracted = []

    for elt in vocab_extracted:
        if elt in token_to_merges:
            merge = token_to_merges[elt]

            assert (merge[0] in vocab_extracted) and (merge[1] in vocab_extracted), merge
            merges_extracted.append(token_to_merges[elt])

    tokenizer_extracted = replace_vocab_and_merges(deepcopy(tokenizer_base), vocab_extracted, merges_extracted)

    vocab_extended = vocab_base.copy()
    merges_extended = merges_base.copy()

    for token in vocab_extracted:
        if token not in vocab_base:
            vocab_extended[token] = len(vocab_extended)
            merges_extended.append(token_to_merges[token])

    tokenizer_extended = replace_vocab_and_merges(deepcopy(tokenizer_base), vocab_extended, merges_extended)

    return vocab_extracted, merges_extracted, tokenizer_extracted, vocab_extended, merges_extended, tokenizer_extended


def sequential_train(
        input_paths: str,
        output_dir: str,
        langs_order: str,
        vocab_size: int,
        tokenizer_base_path: str = None,
        max_token_length: Optional[int] = None,
        max_sentences: int = -1,
        text_field: Optional[str] = None,
        save_steps: bool = True,
):
    """Train the sequential multilingual BPE tokenizer and write its tokenizer
    directory to ``output_dir``. ``vocab_size`` is the per-language budget k. Each input
    path is any user path (plain text, .zst, JSONL via ``text_field``, or a
    directory of such files)."""
    os.makedirs(output_dir, exist_ok=True)

    from modular_tokenizers.config import as_str_list

    langs_order = as_str_list(langs_order)
    input_paths = as_str_list(input_paths)

    assert len(langs_order) == len(input_paths)

    per_lang = {}

    for i, lang in enumerate(langs_order):
        if i == 0:
            if tokenizer_base_path is None:
                logger.info(f"Step 0 ({lang}): training initial tokenizer from scratch, vocab size {vocab_size}")
                vocab_base, merges_base, tokenizer_base = train_bpe_from_scratch(
                    input_paths[i],
                    vocab_size,
                    text_field=text_field,
                )
            else:
                logger.info(f"Step 0 ({lang}): loading base tokenizer at {tokenizer_base_path}")
                from tokenizers import Tokenizer as HFTokenizer
                from transformers import PreTrainedTokenizerFast

                tokenizer_obj = HFTokenizer.from_file(tokenizer_base_path)
                tokenizer_base = PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj)
                vocab_base, merges_base = get_vocab_and_merges(tokenizer_base)

            # Language 1's subtokenizer IS the base tokenizer.
            per_lang[lang] = (vocab_base, merges_base, tokenizer_base)

            if save_steps:
                save_step(tokenizer_base, os.path.join(output_dir, f"sequential/step_0_{lang}"), vocab_base, merges_base)
                save_step(tokenizer_base, os.path.join(output_dir, f"extracted/step_0_{lang}"), vocab_base, merges_base)

            vocab_sequential = vocab_base.copy()
            merges_sequential = merges_base.copy()
            tokenizer_sequential = deepcopy(tokenizer_base)

        else:
            logger.info(f"Step {i} ({lang}): extending tokenizer of size {tokenizer_sequential.vocab_size}")
            new_vocab, new_merges = learn_extension(
                input_paths[i],
                tokenizer_sequential,
                vocab_size,
                max_token_length,
                max_sentences,
                text_field,
            )

            (
                vocab_extracted,
                merges_extracted,
                tokenizer_extracted,
                vocab_sequential,
                merges_sequential,
                tokenizer_sequential,
            ) = compute_extension_and_extraction(
                vocab_sequential,
                merges_sequential,
                new_vocab,
                new_merges,
                vocab_size,
                tokenizer_sequential,
            )

            per_lang[lang] = (vocab_extracted, merges_extracted, tokenizer_extracted)

            if save_steps:
                save_step(tokenizer_sequential, os.path.join(output_dir, f"sequential/step_{i}_{lang}"), vocab_sequential, merges_sequential)
                save_step(tokenizer_extracted, os.path.join(output_dir, f"extracted/step_{i}_{lang}"), vocab_extracted, merges_extracted)

                added_path = os.path.join(output_dir, f"added/step_{i}_{lang}")
                os.makedirs(added_path, exist_ok=True)
                write_json(new_vocab, os.path.join(added_path, "vocab.json"))
                write_json([" ".join(x) for x in new_merges], os.path.join(added_path, "merges.json"))

            logger.info(f"Sequential tokenizer size: {tokenizer_sequential.vocab_size}")
            logger.info(f"Extracted tokenizer size: {tokenizer_extracted.vocab_size}")

    serialization.write(
        output_dir,
        global_tokenizer=tokenizer_sequential,
        global_vocab=vocab_sequential,
        global_merges=merges_sequential,
        per_lang=per_lang,
        langs_order=langs_order,
    )
    serialization.validate(output_dir, langs_order)


class SequentialArgs(BaseModel):
    input_paths: str | None = None
    output_dir: str | None = None
    langs_order: str | None = None
    vocab_size: int | None = None
    tokenizer_base_path: str | None = None
    max_token_length: int | None = None
    max_sentences: int = -1
    text_field: str | None = None
    save_steps: bool = True


def main():
    from modular_tokenizers.config import parse_args_to_pydantic_model, setup_logging

    setup_logging(os.environ.get("LOGLEVEL"))
    args = parse_args_to_pydantic_model(SequentialArgs)
    assert args.input_paths and args.output_dir and args.langs_order and args.vocab_size, \
        "input_paths, output_dir, langs_order and vocab_size are required"
    sequential_train(**args.model_dump())


if __name__ == "__main__":
    main()
