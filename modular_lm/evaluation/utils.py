"""Shared evaluation helpers: batching, reporting, text metrics and masked
sequence scoring."""

import json
import math
import re
import string
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import optax


def batched(iterator, batch_size: int):
    """Yield lists of ``batch_size`` items (the last one may be shorter)."""
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def report_metrics(metrics: dict, run_info: dict, logger, output: str | None):
    """Log one evaluation's metrics (+ its run settings) and append them as a
    JSON line to ``output`` when given."""
    record = dict(metrics) | run_info
    logger.log(**{k: (round(v, 4) if isinstance(v, float) else v)
                  for k, v in record.items()})
    if output:
        with open(output, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=float) + "\n")


# ---------------------------------------------------------------------------
# Text metrics (SQuAD-style normalization)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, golds: list[str]) -> float:
    prediction = normalize_answer(prediction)
    return float(any(prediction == normalize_answer(g) for g in golds))


def _compute_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def f1_score(prediction: str, golds: list[str]) -> float:
    return max(_compute_f1(prediction, g) for g in golds)


# ---------------------------------------------------------------------------
# Masked sequence scoring
# ---------------------------------------------------------------------------

def pad_right(tokens: list[int], pad_id: int, length: int) -> list[int]:
    if len(tokens) < length:
        return list(tokens) + [pad_id] * (length - len(tokens))
    return list(tokens)[:length]


class SequenceScorer:
    """Scores text sequences with a model, restricting the output distribution
    to one subtokenizer's vocabulary (additive logits mask). Sequences are
    right-padded to a multiple of 256 so jit recompiles per length bucket only.
    """

    def __init__(self, model, params, tokenizer):
        self.params = params
        self.tokenizer = tokenizer
        self.is_modular = hasattr(tokenizer, "subtokenizers")

        def _sequence_ce(params, x, y, mask, pad_id):
            logits, _ = model.apply(params, x)
            logits = logits + mask
            labels = jax.nn.one_hot(y, logits.shape[-1])
            ce = optax.softmax_cross_entropy(logits, labels)
            ce = jnp.where(y == pad_id, 0.0, ce)
            return jnp.sum(ce, axis=1)

        def _all_logits(params, x, mask):
            logits, _ = model.apply(params, x)
            return logits + mask

        self._sequence_ce = jax.jit(_sequence_ce)
        self._all_logits = jax.jit(_all_logits)

    def encode(self, text: str, subtokenizer_id: str | None) -> list[int]:
        if self.is_modular:
            return list(self.tokenizer.encode(text, subtokenizer_id, bos=False, eos=False))
        return list(self.tokenizer.encode(text, bos=False, eos=False))

    def _logits_mask(self, subtokenizer_id: str | None) -> jnp.ndarray:
        """Additive (1, 1, vocab) mask of the subtokenizer's vocabulary."""
        if not self.is_modular or subtokenizer_id is None:
            return jnp.zeros((1, 1, self.tokenizer.vocab_size()), dtype=jnp.float32)
        mask_bool = np.asarray(self.tokenizer.mask(subtokenizer_id))
        mask = np.full((1, 1, len(mask_bool)), -np.inf, dtype=np.float32)
        mask[0, 0, mask_bool] = 0.0
        return jnp.asarray(mask)

    def _batch(self, sequences: list[str], subtokenizer_id: str | None):
        tokens = [self.encode(s, subtokenizer_id) for s in sequences]
        last_pos = [len(t) - 1 for t in tokens]
        pad_length = 256 * math.ceil(max(len(t) for t in tokens) / 256)
        x = jnp.asarray([pad_right(t, self.tokenizer.pad_id(), pad_length) for t in tokens])
        return x, last_pos

    def scores(self, sequences: list[str], subtokenizer_id: str | None) -> np.ndarray:
        """Total cross-entropy of each sequence under the subtokenizer's mask."""
        x, _ = self._batch(sequences, subtokenizer_id)
        mask = self._logits_mask(subtokenizer_id)
        ce = self._sequence_ce(self.params, x[:, :-1], x[:, 1:], mask,
                               self.tokenizer.pad_id())
        return np.asarray(ce)

    def next_token_logits(self, sequences: list[str], words: list[str],
                          subtokenizer_id: str | None) -> np.ndarray:
        """Logits of each word's first token, after each sequence's last token."""
        x, last_pos = self._batch(sequences, subtokenizer_id)
        mask = self._logits_mask(subtokenizer_id)
        logits = self._all_logits(self.params, x, mask)
        logits = logits[jnp.arange(x.shape[0]), jnp.asarray(last_pos)]
        word_ids = [self.encode(w, subtokenizer_id)[0] for w in words]
        return np.asarray(logits[:, word_ids])
