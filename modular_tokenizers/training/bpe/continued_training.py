"""Continued BPE training: the per-language engine of sequential training.

Given an existing tokenizer and a new corpus, pre-tokenize the corpus with the
existing tokenizer and run the BPE merge loop *on top of the frozen existing
vocabulary*, returning the new tokens and merges (with their frequencies).
Every unit the new merges build on already exists, so extending the tokenizer
with them keeps all tokens constructible.

Adapted from tokenizer-extension (Purason et al., "Teaching Old Tokenizers New
Words", https://github.com/taidopurason/tokenizer-extension, Apache-2.0 — see
the Licenses section of the README). Changes vs upstream: HF tokenizers only
(the SentencePiece code path was unused and removed).
"""
import logging
from collections import defaultdict
from heapq import heappop, heappush
from typing import Iterable, Optional

from tqdm import tqdm


def group_tokens(text, tokenizer):
    """Pre-tokenize ``text`` into words and segment each word with the
    tokenizer's current BPE model -> list of token tuples."""
    pre_tokenizer = tokenizer._tokenizer.pre_tokenizer
    if pre_tokenizer is None:
        raise ValueError("Tokenizer must have a pre-tokenizer")

    if tokenizer._tokenizer.normalizer is not None:
        text = tokenizer._tokenizer.normalizer.normalize_str(text)

    pre_tokenized = [x[0] for x in pre_tokenizer.pre_tokenize_str(text)]

    grouped_new_words = []
    for word in pre_tokenized:
        group = [token.value for token in tokenizer._tokenizer.model.tokenize(word)]
        if len(group) > 0:
            grouped_new_words.append(group)

    return list(map(tuple, grouped_new_words))


def compute_pair_freqs(splits, word_freqs):
    pair_freqs = defaultdict(int)
    where_to_update = defaultdict(set)

    for word, freq in word_freqs.items():
        split = splits[word]
        # Single-token words count as "unigram" candidates: sequential training
        # compares the most frequent existing unit against the most frequent
        # pair (paper Algorithm 1), so whole-word occurrences must be tracked.
        if len(split) == 1:
            pair_freqs[split] += freq
            where_to_update[split].add(word)
            continue

        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
            where_to_update[pair].add(word)
    return pair_freqs, where_to_update


def merge_pair(a, b, splits, word_freqs, pair_freqs, queue, where_to_update):
    updated_pairs = defaultdict(int)

    for word in list(where_to_update[(a, b)]):
        freq = word_freqs[word]
        split = splits[word]
        if len(split) == 1:
            continue

        split = list(split)
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                new_token = a + b
                split = split[:i] + [new_token] + split[i + 2:]
            else:
                i += 1

        prev_pairs = list(zip(splits[word][:-1], splits[word][1:]))
        splits[word] = split
        new_pairs = list(zip(splits[word][:-1], splits[word][1:]))
        for pair in set(new_pairs) - set(prev_pairs):
            where_to_update[pair].add(word)
        for pair in set(prev_pairs) - set(new_pairs):
            where_to_update[pair].discard(word)
        for pair in prev_pairs:
            updated_pairs[pair] -= freq
        for pair in new_pairs:
            updated_pairs[pair] += freq

    for pair, change in updated_pairs.items():
        new_freq = pair_freqs.get(pair, 0) + change
        pair_freqs[pair] = new_freq
        if new_freq > 0:
            heappush(queue, (-new_freq, pair))

    del pair_freqs[(a, b)]
    del where_to_update[(a, b)]
    return splits


def train_vocab_extension(
        tokenizer,
        corpus: Iterable[str],
        extension_size: int,
        max_token_length: Optional[int] = None,
) -> dict:
    """
    :param tokenizer: The tokenizer to continually train (PreTrainedTokenizerFast)
    :param corpus: Training corpus
    :param extension_size: The number of tokens to add to the vocabulary
    :param max_token_length: maximum length of a token created
    :return: Dictionary with keys: 'vocab', 'merges', 'pair_freqs', 'word_freqs'
    """
    split_freqs = defaultdict(int)

    for text in tqdm(corpus, desc="computing frequencies", mininterval=1):
        grouped_tokens = group_tokens(text, tokenizer)
        for word in grouped_tokens:
            split_freqs[word] += 1

    splits = {"".join(split): split for split in split_freqs}
    word_freqs = {"".join(split): freq for split, freq in split_freqs.items()}
    pair_freqs, where_to_update = compute_pair_freqs(splits, word_freqs)

    pair_queue = []
    for pair, freq in pair_freqs.items():
        heappush(pair_queue, (-freq, pair))

    vocab_size = extension_size
    vocab = {}
    merges = []

    with tqdm(total=vocab_size, desc="training") as pbar:
        while len(vocab) < vocab_size:
            if not pair_queue:
                logging.getLogger(__name__).warning(
                    f"corpus exhausted after {len(vocab)}/{vocab_size} new tokens; stopping early"
                )
                break
            max_freq, best_pair = heappop(pair_queue)
            max_freq = -max_freq  # Convert back to positive

            # Skip stale entries
            if best_pair is None or pair_freqs.get(best_pair, None) != max_freq:
                continue

            new_token = "".join(best_pair)
            if len(best_pair) == 1:
                del pair_freqs[best_pair]
                del where_to_update[best_pair]

            else:
                if max_token_length is not None and len(new_token) > max_token_length:
                    del pair_freqs[best_pair]
                    del where_to_update[best_pair]
                    continue

                splits = merge_pair(*best_pair, splits, word_freqs, pair_freqs, pair_queue, where_to_update)

            merges.append(best_pair)

            if new_token not in vocab:
                vocab[new_token] = len(vocab)

            if len(vocab) % 100 == 0 or len(vocab) == vocab_size:
                pbar.n = len(vocab)
                pbar.refresh()

    return {"vocab": vocab, "merges": merges, "pair_freqs": pair_freqs, "word_freqs": word_freqs}
