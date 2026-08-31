"""Normalized Sequence Length (NSL), Eq. 1 of the paper.

NSL of tokenizer T on language L = (total tokens of T over L's eval texts) /
(total tokens of a monolingual reference tokenizer over the same texts).
NSL close to 1 means T compresses L as well as the monolingual reference
(paper Table 1: FLORES-200 devtest, 24k monolingual BPE references).

Inputs are explicit per-language mappings — no directory organization is
assumed. Each data path may be a plain text file, a .zst file, a JSONL file
(use ``text_field``), or a directory of such files. The languages evaluated
are the keys of ``data``.

CLI (dotted keys per language, or put them in a YAML passed as config=...):
    python -m modular_tokenizers.evaluation.nsl \
        data.bg=/my/eval/bulgarian.txt data.el=/elsewhere/greek.jsonl text_field=text \
        tokenizer.type=modular tokenizer.path=/path/to/tokenizer_dir/ \
        tokenizer.langs=bg,el \
        tokenizer.extraction_strategy.strategy=merged_seq_bpe \
        reference.type=base \
        reference_paths.bg=/refs/bg_bpe24k.json reference_paths.el=/refs/el_bpe24k.json \
        [composition=all] [save_path=nsl.json]

(The tokenizer's algorithm/backend are detected from the file; a tokenizer
directory provides its lang_to_ids.json automatically.)

By default a modular tokenizer is evaluated with each language's own
subtokenizer; pass ``composition=all`` (or any composition id) to override.
"""
import json
import logging
import os

from pydantic import BaseModel

from modular_tokenizers.config import parse_args_to_pydantic_model, setup_logging
from modular_tokenizers.subtokenizers.modular_tokenizer import TokenizerArgs
from modular_tokenizers.training.utils import iter_docs

logger = logging.getLogger(__name__)

def _is_modular(tok_type: str) -> bool:
    return tok_type != "base"


class NSLArgs(BaseModel):
    data: dict[str, str] = {}             # {lang: eval-data path}
    text_field: str | None = None         # extract this field from JSONL lines
    tokenizer: TokenizerArgs = TokenizerArgs()
    reference: TokenizerArgs = TokenizerArgs()   # shared reference settings (type, ...)
    reference_paths: dict[str, str] = {}  # {lang: reference tokenizer file}
    composition: str | None = None        # default: the language itself
    max_load: int = -1
    save_path: str | None = None


def _n_tokens(tokenizer, tok_type: str, texts: list[str], composition: str) -> int:
    total = 0
    for text in texts:
        if _is_modular(tok_type):
            total += len(tokenizer.encode(text, composition, bos=False, eos=False))
        else:
            total += len(tokenizer.encode(text, bos=False, eos=False))
    return total


def compute_nsl(args: NSLArgs) -> dict:
    langs = list(args.data.keys())
    assert langs, "data must map at least one language to an eval-data path"
    missing = [l for l in langs if l not in args.reference_paths]
    assert not missing, f"reference_paths missing for languages: {missing}"

    tokenizer = args.tokenizer.build_tokenizer()

    results = {}
    for lang in langs:
        texts = list(iter_docs(args.data[lang], args.text_field, args.max_load))

        reference_args = args.reference.model_copy(update={"path": args.reference_paths[lang]})
        reference = reference_args.build_tokenizer()

        composition = args.composition or lang
        n = _n_tokens(tokenizer, args.tokenizer.type, texts, composition)
        n_ref = _n_tokens(reference, args.reference.type, texts, composition)

        results[lang] = {"nsl": n / n_ref, "tokens": n, "tokens_ref": n_ref, "texts": len(texts)}
        logger.info(f"{lang}: NSL={results[lang]['nsl']:.3f} ({n} / {n_ref} tokens, {len(texts)} texts)")

    return results


def main():
    args = parse_args_to_pydantic_model(NSLArgs)
    results = compute_nsl(args)

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        with open(args.save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"saved -> {args.save_path}")

    print(json.dumps({l: round(r["nsl"], 4) for l, r in results.items()}, indent=2))


if __name__ == "__main__":
    setup_logging(os.environ.get("LOGLEVEL"))
    main()
