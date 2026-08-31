"""Merging monolingual Unigram tokenizers (paper §3.1).

The merged vocabulary is the union of the monolingual vocabularies (first
tokenizer's pieces keep their positions; new pieces are appended in order).
Token probabilities can then be re-estimated over the combined corpus with
``em_reestimate``.

Both backends are supported and auto-detected from the file extension:
``.model`` files (SentencePiece, the paper backend) or ``.json`` files
(HF `tokenizers`).

CLI (writes the tokenizer directory: tokenizer.model|tokenizer.json + lang_to_ids.json):
    python -m modular_tokenizers.training.unigram.merge \
        model_paths=bg.model,cs.model,... langs=bg,cs,... output_dir=out/
"""
import logging

import sentencepiece
from pydantic import BaseModel
from sentencepiece import SentencePieceProcessor, sentencepiece_model_pb2

from modular_tokenizers.training.unigram import serialization

logger = logging.getLogger(__name__)


def merge_tokenizers(tokenizers):
    """Union of the vocabularies of SentencePiece processors, returned as a
    new processor. Piece scores are kept as-is (un-normalized merge)."""
    protos = {}
    for i in range(len(tokenizers)):
        protos[i] = sentencepiece_model_pb2.ModelProto()
        protos[i].ParseFromString(tokenizers[i].serialized_model_proto())

    set_pieces = set(p.piece for p in protos[0].pieces)

    for i in range(1, len(tokenizers)):
        for piece in protos[i].pieces:
            if piece.piece not in set_pieces:
                protos[0].pieces.append(piece)
                set_pieces.add(piece.piece)

    tokenizer_new = sentencepiece.SentencePieceProcessor()
    tokenizer_new.LoadFromSerializedProto(protos[0].SerializeToString())
    return tokenizer_new


def merge_hf_tokenizers(tokenizers):
    """Union of the vocabularies of HF `tokenizers` Unigram tokenizers (same
    semantics as ``merge_tokenizers``: first-seen piece keeps its score)."""
    from modular_tokenizers.training.utils import get_hf_json, hf_from_json

    cfg = get_hf_json(tokenizers[0])
    assert cfg["model"]["type"] == "Unigram"
    merged_vocab = list(cfg["model"]["vocab"])
    seen = {p for p, _ in merged_vocab}

    for tok in tokenizers[1:]:
        other = get_hf_json(tok)
        assert other["model"]["type"] == "Unigram"
        cfg["model"]["byte_fallback"] = cfg["model"].get("byte_fallback", False) or other["model"].get("byte_fallback", False)
        for piece, score in other["model"]["vocab"]:
            if piece not in seen:
                merged_vocab.append([piece, score])
                seen.add(piece)

    cfg["model"]["vocab"] = merged_vocab
    return hf_from_json(cfg)


def merge_model_files(model_paths: str, langs: str, output_dir: str):
    """Merge monolingual tokenizer files and write the tokenizer directory to
    ``output_dir`` — see ``training/unigram/serialization.py``. The backend is
    detected from the extensions (.model = SentencePiece, .json = HF)."""
    from modular_tokenizers.config import as_str_list

    paths = as_str_list(model_paths)
    lang_list = as_str_list(langs)
    assert len(paths) == len(lang_list), "model_paths must match langs"

    is_hf = paths[0].endswith(".json")
    assert all(p.endswith(".json") == is_hf for p in paths), "all model files must use the same backend"

    if is_hf:
        from tokenizers import Tokenizer as HFTokenizer

        tokenizers = {lang: HFTokenizer.from_file(p) for lang, p in zip(lang_list, paths)}
        merged = merge_hf_tokenizers([tokenizers[l] for l in lang_list])

        piece_to_id = merged.get_vocab()
        lang_to_ids = {}
        for lang, tok in tokenizers.items():
            own_vocab = sorted(tok.get_vocab().items(), key=lambda x: x[1])
            lang_to_ids[lang] = [piece_to_id[piece] for piece, _ in own_vocab]

        vocab_size = merged.get_vocab_size()
        serialization.write(
            output_dir,
            lang_to_ids=lang_to_ids,
            merged_hf_tokenizer=merged,
        )
    else:
        processors = {lang: SentencePieceProcessor(model_file=p) for lang, p in zip(lang_list, paths)}
        merged = merge_tokenizers([processors[l] for l in lang_list])

        piece_to_id = {merged.IdToPiece(i): i for i in range(merged.vocab_size())}
        lang_to_ids = {
            lang: [piece_to_id[proc.IdToPiece(i)] for i in range(proc.vocab_size())]
            for lang, proc in processors.items()
        }

        vocab_size = merged.vocab_size()
        serialization.write(
            output_dir,
            lang_to_ids=lang_to_ids,
            merged_model_proto=merged.serialized_model_proto(),
        )

    serialization.validate(output_dir, lang_list)
    logger.info(f"Merged {len(paths)} tokenizers (|vocab|={vocab_size})")


class MergeArgs(BaseModel):
    model_paths: str | None = None
    langs: str | None = None
    output_dir: str | None = None


def main():
    import os

    from modular_tokenizers.config import parse_args_to_pydantic_model, setup_logging

    setup_logging(os.environ.get("LOGLEVEL"))
    args = parse_args_to_pydantic_model(MergeArgs)
    assert args.model_paths and args.langs and args.output_dir, \
        "model_paths, langs and output_dir are required"
    merge_model_files(**args.model_dump())


if __name__ == "__main__":
    main()
