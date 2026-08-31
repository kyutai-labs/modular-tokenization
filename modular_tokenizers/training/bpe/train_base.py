"""From-scratch BPE training (HF `tokenizers`).

Used for (i) the first language of sequential training, (ii) monolingual
reference tokenizers (NSL references), and (iii) the joint multilingual
baseline — pass the concatenated corpora of all languages and the full
vocabulary budget (e.g. 384k for 21 languages) instead of a per-language one.

CLI (key=value; a YAML can be passed as config=file.yaml):
    python -m modular_tokenizers.training.bpe.train_base \
        input_path=a.txt,b.txt vocab_size=24000 \
        [output_dir=out/] [tokenizer_json_path=tok.json]
"""
import logging
import os

from tokenizers import Regex, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
from tokenizers.trainers import BpeTrainer
from pydantic import BaseModel
from transformers import PreTrainedTokenizerFast

from modular_tokenizers.config import parse_args_to_pydantic_model, setup_logging

from modular_tokenizers.subtokenizers.extraction import get_vocab_and_merges
from modular_tokenizers.training.utils import SPECIAL_TOKENS

logger = logging.getLogger(__name__)

# NOTE: the default ByteLevel(use_regex=True) applies the GPT-2 regex whose letter
# class \p{L}+ EXCLUDES Unicode combining marks (\p{M}). For Brahmic/Indic scripts
# (Devanagari, Sinhala, Telugu, ...) this cuts every consonant+matra/virama cluster
# apart before BPE runs, so those merges never form and compression collapses
# (e.g. Nepali went from ~0.25 to ~0.69 tokens/char). We replace it with the same
# GPT-2-style pre-tokenization but grouping letters WITH their following marks; on
# scripts without combining marks (Latin/Cyrillic/Greek) this is byte-identical.
# This base pre-tokenizer propagates to every subsequent language via
# train_vocab_extension, which reads the tokenizer's pre_tokenizer.
MARK_AWARE_PRETOKENIZER_REGEX = r" ?[\p{L}\p{M}]+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+"


def train_bpe_from_scratch(
        input_path: str,
        vocab_size: int,
        output_dir: str | None = None,
        tokenizer_json_path: str | None = None,
        text_field: str | None = None,
):
    """Train a byte-level BPE tokenizer with the mark-aware pre-tokenizer.

    ``input_path`` is a comma-separated list of sources; each source is any
    user path — a plain text file, a .zst file, a JSONL file (use
    ``text_field``), or a directory of such files. Returns
    ``(vocab, merges, tokenizer)`` with the tokenizer wrapped in a
    ``PreTrainedTokenizerFast``. Optionally saves ``save_pretrained`` files to
    ``output_dir`` and/or the raw ``tokenizer.json`` to ``tokenizer_json_path``.
    """
    from modular_tokenizers.config import as_str_list
    from modular_tokenizers.training.utils import iter_docs

    files = as_str_list(input_path)
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
    )

    tokenizer.normalizer = None
    tokenizer.pre_tokenizer = Sequence([
        Split(Regex(MARK_AWARE_PRETOKENIZER_REGEX), behavior="isolated"),
        ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer.decoder = ByteLevelDecoder()

    logger.info(f"Training BPE from scratch: vocab_size={vocab_size} on {len(files)} source(s)")
    plain_text_files = all(
        not os.path.isdir(f) and not f.endswith(".zst") for f in files
    ) and text_field is None
    if plain_text_files:
        tokenizer.train(files, trainer)
    else:
        tokenizer.train_from_iterator(
            (doc for f in files for doc in iter_docs(f, text_field)), trainer
        )

    if tokenizer_json_path is not None:
        os.makedirs(os.path.dirname(tokenizer_json_path) or ".", exist_ok=True)
        tokenizer.save(tokenizer_json_path)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )

    if output_dir is not None:
        tokenizer.save_pretrained(output_dir)

    vocab, merges = get_vocab_and_merges(tokenizer)

    return vocab, merges, tokenizer


class TrainBaseArgs(BaseModel):
    input_path: str | None = None
    vocab_size: int | None = None
    output_dir: str | None = None
    tokenizer_json_path: str | None = None
    text_field: str | None = None


def main():
    args = parse_args_to_pydantic_model(TrainBaseArgs)
    assert args.input_path and args.vocab_size, "input_path and vocab_size are required"
    train_bpe_from_scratch(**args.model_dump())


if __name__ == "__main__":
    setup_logging(os.environ.get("LOGLEVEL"))
    main()
