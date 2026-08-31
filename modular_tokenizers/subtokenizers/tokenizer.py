"""A uniform tokenizer wrapper: one interface over SentencePiece and HF
`tokenizers` backends. The backend is inferred: from the file extension
(``.json`` -> HF, anything else -> SentencePiece), or from which in-memory
tokenizer object is given.
"""
import re
from typing import Optional

import numpy as np
from sentencepiece import SentencePieceProcessor
from tokenizers import Tokenizer as HFTokenizer


class Tokenizer:
    """Uniform wrapper over a SentencePiece or HF `tokenizers` tokenizer.

    The backend is inferred: from the file extension (``.json`` -> HF,
    anything else -> SentencePiece), or from which in-memory tokenizer
    object is given.
    """

    def __init__(
        self,
        *,
        path: Optional[str] = None,
        spm_tokenizer: Optional[SentencePieceProcessor] = None,
        hf_tokenizer: Optional[HFTokenizer] = None,
    ):
        if hf_tokenizer is not None:
            use_hf = True
        elif spm_tokenizer is not None:
            use_hf = False
        else:
            assert path is not None
            use_hf = path.endswith(".json")
        self.use_hf = use_hf

        if use_hf:
            if path is not None:
                self.tokenizer = HFTokenizer.from_file(path)
            else:
                assert hf_tokenizer is not None
                self.tokenizer = hf_tokenizer

            vocab = self.tokenizer.get_vocab()
            vocab = dict(sorted(((tok, idx) for tok, idx in vocab.items()), key=lambda x: x[1]))
            self.pieces_to_id_mapping = vocab
            self.id_to_pieces_mapping = {v: k for k, v in vocab.items()}

            self._bos_id = vocab.get("<s>")
            self._eos_id = vocab.get("</s>")
            self._pad_id = vocab.get("<pad>")
            self._unk_id = vocab.get("<unk>")

        else:
            if path is not None:
                self.tokenizer = SentencePieceProcessor(model_file=path)
            else:
                assert spm_tokenizer is not None
                self.tokenizer = spm_tokenizer

            self.id_to_pieces_mapping = {}
            self.pieces_to_id_mapping = {}

            for i in range(self.tokenizer.vocab_size()):
                piece = self.tokenizer.IdToPiece(i)
                self.id_to_pieces_mapping[i] = piece
                self.pieces_to_id_mapping[piece] = i

    def bos_id(self):
        return self._bos_id if self.use_hf else self.tokenizer.bos_id()

    def eos_id(self):
        return self._eos_id if self.use_hf else self.tokenizer.eos_id()

    def pad_id(self):
        return self._pad_id if self.use_hf else self.tokenizer.pad_id()

    def unk_id(self):
        return self._unk_id if self.use_hf else self.tokenizer.unk_id()

    def vocab_size(self):
        return self.tokenizer.get_vocab_size() if self.use_hf else self.tokenizer.vocab_size()

    def encode(self, text, bos=True, eos=True):
        if self.use_hf:
            tokens = self.tokenizer.encode(text).ids
        else:
            tokens = self.tokenizer.encode(text)

        if bos:
            tokens = [self.bos_id()] + tokens
        if eos:
            tokens = tokens + [self.eos_id()]

        return np.array(tokens, dtype=np.int32)

    def decode(self, tokens, strip_bos=False, strip_eos=False):
        tokens = [int(i) for i in tokens if i != self.pad_id() and i != self.unk_id()]

        if strip_bos and self.bos_id() in tokens[1:]:
            tokens = tokens[:tokens.index(self.bos_id(), 1)]

        if strip_eos and self.eos_id() in tokens[1:]:
            tokens = tokens[:tokens.index(self.eos_id(), 1)]

        return self.tokenizer.decode(tokens)

    BYTE_TOKEN_RE = re.compile(r"<0x[0-9A-Fa-f]{2}>")

    def is_special_token(self, i):
        if self.use_hf:
            token = self.id_to_pieces_mapping.get(i)
            if token is None:
                return False
            return token in {"<s>", "</s>", "<pad>", "<unk>"} or bool(self.BYTE_TOKEN_RE.fullmatch(token))

        return any([
            self.tokenizer.IsByte(i),
            self.tokenizer.IsControl(i),
            self.tokenizer.IsUnknown(i),
            self.tokenizer.IsUnused(i),
        ])
