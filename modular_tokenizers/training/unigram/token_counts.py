"""Per-language token occurrence counts over a corpus.

Feeds the frequency-based extraction strategy (``ExtractionConfig(strategy=
"common", path_tokens_occurence=...)``): a language's subtokenizer keeps the
top-k most frequent tokens of the global tokenizer on that language's data.

``data_path`` is any user path — a plain text file, a .zst file, a JSONL file
(use ``text_field``), or a directory of such files.

CLI:
    python -m modular_tokenizers.training.unigram.token_counts \
        data_path=/any/path/to/fr_data save_path=counts/fr text_field=text \
        tokenizer.type=base tokenizer.path=<tokenizer_dir>/tokenizer.model
"""
import json
import logging
import os

import sentencepiece
from pydantic import BaseModel

from modular_tokenizers.config import parse_args_to_pydantic_model, setup_logging
from modular_tokenizers.subtokenizers.modular_tokenizer import TokenizerArgs
from modular_tokenizers.training.utils import iter_docs

logger = logging.getLogger(__name__)


class CountArgs(BaseModel):
    data_path: str | None = None
    save_path: str | None = None
    text_field: str | None = None
    nb_ex: int = 10_000_000
    tokenizer: TokenizerArgs = TokenizerArgs()


def save_counts(count, n, save_path):
    path = os.path.join(save_path, f'{n // 1000}k.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as f:
        json.dump(count, f, indent=4)


def main():
    args = parse_args_to_pydantic_model(CountArgs)

    if args.tokenizer.type == 'base':
        tokenizer = sentencepiece.SentencePieceProcessor(model_file=args.tokenizer.path)
    else:
        tokenizer = args.tokenizer.build_tokenizer()

    logger.info(json.dumps(args.model_dump()))
    logger.info('Start')

    count = {i: 0 for i in range(tokenizer.vocab_size())}

    n = 0
    for text in iter_docs(args.data_path, args.text_field, args.nb_ex):
        if args.tokenizer.type == 'base':
            tokens = tokenizer.encode(text)
        else:
            # Counts are over the GLOBAL vocabulary: tokenize with the full
            # composition.
            tokens = tokenizer.encode(text, 'all', bos=False, eos=False)

        for j in tokens:
            count[j] += 1

        n += 1

        if n % 100000 == 0:
            logger.info(f'{n}')
            save_counts(count, n, args.save_path)

    save_counts(count, n, args.save_path)


if __name__ == "__main__":
    setup_logging(os.environ.get("LOGLEVEL"))
    main()
