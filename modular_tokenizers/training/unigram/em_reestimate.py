"""EM re-estimation of Unigram token probabilities over a combined corpus
(paper §3.1, Algorithm unigram_em — the MERGED_NORM setting).

Given the union vocabulary of monolingual tokenizers (or an explicit vocab
file) and text from the combined corpus, run fixed-vocabulary EM
(forward-backward over all segmentations) and write per-iteration scores.
Optionally apply the final scores back into a merged .model file, producing
the re-estimated merged Unigram tokenizer.

CLI:
    python -m modular_tokenizers.training.unigram.em_reestimate \
        data_path=bg.txt,cs.txt,... tokenizers_path=bg.model,cs.model,... \
        save_path=out/em max_iters=10 \
        [apply_to_model=out/tokenizer.model out_model=out/tokenizer.model]
"""
import json
import logging
import math
import os
import random
import re
from collections import defaultdict
from timeit import default_timer as timer

import sentencepiece
from pydantic import BaseModel
from sentencepiece import sentencepiece_model_pb2

from modular_tokenizers.config import parse_args_to_pydantic_model, setup_logging

logger = logging.getLogger(__name__)


def is_special_token(tokenizer, i):
    return any([
        tokenizer.IsByte(i),
        tokenizer.IsControl(i),
        tokenizer.IsUnknown(i),
        tokenizer.IsUnused(i),
    ])


HF_SPECIAL_RE = re.compile(r"<0x[0-9A-Fa-f]{2}>|<unused\d+>")
HF_SPECIAL_PIECES = {"<s>", "</s>", "<pad>", "<unk>"}


def is_special_piece(piece: str) -> bool:
    """HF counterpart of ``is_special_token`` (byte/control/unused pieces)."""
    return piece in HF_SPECIAL_PIECES or bool(HF_SPECIAL_RE.fullmatch(piece))


def save_scores(scores, it, save_path):
    path = os.path.join(save_path, f'iter={it}.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)


def logsumexp(a, b):
    """stable log(exp(a)+exp(b))"""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))


class TrieNode:
    __slots__ = ("children", "token")

    def __init__(self):
        self.children = dict()
        self.token = None


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, token):
        node = self.root
        for ch in token:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.token = token

    def matches_from(self, s, i):
        """Yield (token, j) for tokens matching s[i:j]."""
        node = self.root
        n = len(s)
        j = i
        while j < n:
            ch = s[j]
            if ch not in node.children:
                break
            node = node.children[ch]
            j += 1
            if node.token is not None:
                yield (node.token, j)


def train_unigram_fixed_vocab(vocab_tokens, texts, max_iters, tol, save_path, verbose=False):
    """
    vocab_tokens: list of token strings (must cover characters or include fallback)
    """

    trie = Trie()
    for t in vocab_tokens:
        trie.insert(t)

    texts_matches = {}
    sentences_skipped = 0

    for id, s in enumerate(texts):
        texts_matches[id] = {}
        for i in range(len(s)):
            matches = [(token, j) for token, j in trie.matches_from(s, i)]
            texts_matches[id][i] = matches
            if len(matches) == 0:
                sentences_skipped += 1

        if verbose and (id + 1) % 100000 == 0:
            logger.info(f'Computed match for {id + 1} sentences')

    if verbose:
        logger.info(f'Computed match for {len(texts)} sentences')
        logger.info(f'Skipped: {sentences_skipped} sentences')

    V = list(vocab_tokens)
    K = len(V)
    logp = {t: -math.log(K) for t in V}

    t_start = timer()
    for it in range(1, max_iters + 1):
        if verbose:
            logger.info(f"EM iter {it} ...")

        counts = defaultdict(float)
        total_sentence_logprob = 0.0
        sentences_seen = 0
        t0 = timer()

        for id, s in enumerate(texts):
            sentences_seen += 1
            n = len(s)

            alpha_log = [-math.inf] * (n + 1)
            alpha_log[0] = 0.0

            for i in range(n):
                if alpha_log[i] == -math.inf:
                    continue

                matched = False
                for token, j in texts_matches[id][i]:
                    matched = True
                    val = alpha_log[i] + logp[token]
                    alpha_log[j] = logsumexp(alpha_log[j], val)
                if not matched:
                    alpha_log = None
                    break

            if alpha_log is None or alpha_log[n] == -math.inf:
                continue

            Z_log = alpha_log[n]
            total_sentence_logprob += Z_log

            beta_log = [-math.inf] * (n + 1)
            beta_log[n] = 0.0

            for i in range(n - 1, -1, -1):
                for token, j in texts_matches[id][i]:
                    val = logp[token] + beta_log[j]
                    beta_log[i] = logsumexp(beta_log[i], val)

            for i in range(n):
                for token, j in texts_matches[id][i]:
                    log_contrib = alpha_log[i] + logp[token] + beta_log[j] - Z_log
                    contrib = math.exp(log_contrib)
                    counts[token] += contrib

            if sentences_seen % 100000 == 0:
                if verbose:
                    logger.info(f"sentences seen: {sentences_seen}, time: {timer() - t0}")
                    t0 = timer()

        Z = sum(counts.values())
        if Z == -math.inf:
            raise ValueError("No counts collected; vocab may not cover corpus. Check vocab or include single-char tokens.")

        new_logp = {t: math.log1p(counts[t]) - math.log1p(Z) for t in V}

        max_change = 0.0
        nb_over = 0
        for t in V:
            diff = abs(new_logp[t] - logp[t])
            nb_over += (diff >= tol)
            if diff > max_change:
                max_change = diff

        logp = new_logp

        if verbose:
            logger.info(
                f"total log-likelihood (sum): {total_sentence_logprob:.4f} | "
                f"Max change in p: {max_change:.6f} | Nb over: {nb_over} | "
                f"Time iter: {(timer() - t_start):.4f}"
            )
            t_start = timer()

        save_scores(new_logp, it, save_path)

        if max_change < tol:
            if verbose:
                logger.info("Converged.")
            break

    return logp


def select_texts(paths, max_input_sentences, max_sentence_length, normalization_func, seed,
                 text_field=None, verbose=False):
    """Load and interleave sentences from the per-language corpora (uniform
    random source order, matching the original data mixing). Each path is any
    user path (plain text, .zst, JSONL via ``text_field``, or a directory of
    such files)."""
    from modular_tokenizers.training.utils import iter_docs

    rng = random.Random(seed)
    sources = {i: iter_docs(path, text_field) for i, path in enumerate(paths.split(','))}

    if verbose:
        logger.info('Loading texts')

    texts = []
    id_source = rng.choice(list(sources.keys()))
    nb_long = 0
    log_flag = False
    while sources:
        try:
            text = next(sources[id_source])
            if text and len(text) <= max_sentence_length:
                text = normalization_func(text)
                texts.append(text)
                if 0 < max_input_sentences <= len(texts):
                    break
                id_source = rng.choice(list(sources.keys()))
                log_flag = True
            elif len(text) > max_sentence_length:
                nb_long += 1

        except StopIteration:
            del sources[id_source]
            if sources:
                id_source = rng.choice(list(sources.keys()))

        if verbose and log_flag and len(texts) % 100000 == 0:
            logger.info(f'Loaded and Normalized {len(texts)} sentences.')
            log_flag = False

    if verbose:
        logger.info(f'Loaded and Normalized {len(texts)} sentences.')
        logger.info(f'Skipped {nb_long} sentences. Too long')

    return texts


def compute_vocab(vocab_files: str | None, tokenizers_path: str | None):
    vocab = []

    if vocab_files is not None:
        for path in vocab_files.split(','):
            with open(path, "r") as file:
                for line in file:
                    line = line.strip()
                    if line:
                        vocab.append(line)

    elif tokenizers_path.split(',')[0].endswith('.json'):
        from tokenizers import Tokenizer as HFTokenizer

        for path in tokenizers_path.split(','):
            tok = HFTokenizer.from_file(path)
            pieces = sorted(tok.get_vocab().items(), key=lambda x: x[1])
            vocab.extend([piece for piece, _ in pieces if not is_special_piece(piece)])

    else:
        tokenizers = {i: sentencepiece.SentencePieceProcessor(model_file=path) for i, path in enumerate(tokenizers_path.split(','))}

        for i in tokenizers:
            vocab.extend([tokenizers[i].id_to_piece(k) for k in range(tokenizers[i].vocab_size()) if not is_special_token(tokenizers[i], k)])

    return vocab


def normalize_text(text):
    return '▁' + text.replace(' ', '▁')


def apply_scores_to_model(model_path: str, scores: dict, out_path: str):
    """Write a copy of a merged model whose piece scores are replaced by the
    EM-re-estimated log-probabilities (special/byte pieces untouched). Handles
    both backends: .model (SentencePiece) and .json (HF `tokenizers`)."""
    if model_path.endswith(".json"):
        from tokenizers import Tokenizer as HFTokenizer

        tok = HFTokenizer.from_file(model_path)
        cfg = json.loads(tok.to_str())
        replaced = 0
        for entry in cfg["model"]["vocab"]:
            piece = entry[0]
            if not is_special_piece(piece) and piece in scores:
                entry[1] = scores[piece]
                replaced += 1
        HFTokenizer.from_str(json.dumps(cfg)).save(out_path)
        logger.info(f"Applied EM scores to {replaced}/{len(cfg['model']['vocab'])} pieces -> {out_path}")
        return

    proc = sentencepiece.SentencePieceProcessor(model_file=model_path)
    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString(proc.serialized_model_proto())

    replaced = 0
    for i, piece in enumerate(proto.pieces):
        if not is_special_token(proc, i) and piece.piece in scores:
            piece.score = scores[piece.piece]
            replaced += 1

    with open(out_path, "wb") as f:
        f.write(proto.SerializeToString())
    logger.info(f"Applied EM scores to {replaced}/{len(proto.pieces)} pieces -> {out_path}")


class EMArgs(BaseModel):
    data_path: str | None = None
    save_path: str | None = None
    vocab_files: str | None = None
    tokenizers_path: str | None = None
    apply_to_model: str | None = None
    out_model: str | None = None
    max_input_sentences: int = 0
    max_iters: int | None = None
    tol: float = 1e-4
    verbose: bool = False
    seed: int = 1234
    max_sentence_length: int = 4192
    text_field: str | None = None


def main():
    args = parse_args_to_pydantic_model(EMArgs)
    assert (args.vocab_files is None) != (args.tokenizers_path is None)
    logger.info(json.dumps(args.model_dump()))

    random.seed(args.seed)

    vocab = compute_vocab(args.vocab_files, args.tokenizers_path)

    if args.tokenizers_path is not None and not args.tokenizers_path.split(',')[0].endswith('.json'):
        tokenizer = sentencepiece.SentencePieceProcessor(model_file=args.tokenizers_path.split(',')[0])
        normalization_func = tokenizer.normalize
    else:
        # HF backend uses the Metaspace convention, which normalize_text mirrors.
        normalization_func = normalize_text

    texts = select_texts(
        args.data_path, args.max_input_sentences, args.max_sentence_length,
        normalization_func, args.seed, text_field=args.text_field, verbose=args.verbose,
    )

    scores = train_unigram_fixed_vocab(
        vocab, texts, args.max_iters, args.tol, args.save_path, verbose=args.verbose
    )

    if args.apply_to_model is not None:
        assert args.out_model is not None, "out_model is required with apply_to_model"
        apply_scores_to_model(args.apply_to_model, scores, args.out_model)


if __name__ == "__main__":
    setup_logging(os.environ.get("LOGLEVEL"))
    main()
