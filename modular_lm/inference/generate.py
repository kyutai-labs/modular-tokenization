"""Text generation with a trained model, on any subtokenizer.

The prompt is encoded with ``subtokenizer_id`` and generation is restricted to
``subtokenizer_id_out``'s vocabulary through an additive logits mask (they
usually coincide; they may differ, e.g. an English prompt continued under a
unified subtokenizer). Token ids are global at every step, so any subtokenizer
can decode what another one prompted.

Run as ``python -m modular_lm.inference.generate run_dir=... prompts=...``
(one prompt per line), or without ``prompts`` for an interactive loop.
"""

import json
import math
from timeit import default_timer as timer

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from pydantic import BaseModel, ConfigDict

from modular_lm.training.checkpoint import load_model
from modular_tokenizers.config import parse_args_to_pydantic_model


def pad_left(tokens: list[int], pad_id: int, length: int) -> list[int]:
    if len(tokens) < length:
        return [pad_id] * (length - len(tokens)) + tokens
    return tokens[:length]


def sample_top_n(
    logits: Array, key: Array, n: int = 25, temperature: float = 0.8
) -> Array:
    idx = jnp.argsort(-logits, axis=-1)
    top_n_logits = jnp.take_along_axis(logits, idx[..., :n], axis=-1)
    pred = jax.random.categorical(key, top_n_logits / temperature)
    pred = jnp.expand_dims(pred, axis=-1)
    pred = jnp.take_along_axis(idx, pred, axis=-1)
    return jnp.squeeze(pred, axis=-1)


def trim_cache(
    cache: dict[str, jax.Array | tuple[jax.Array, jax.Array]],
) -> dict[str, jax.Array | tuple[jax.Array, jax.Array]]:
    """Drop the oldest cache position, keeping the window length constant so
    the generation loop stays a fixed-shape ``fori_loop`` body."""
    new_cache = {}
    for k, v in cache.items():
        if k == "tokens":
            new_cache[k] = v[:, 1:]
        else:
            new_cache[k] = v[0][:, 1:, ...], v[1][:, 1:, ...]
    return new_cache


class Generator:
    def __init__(self, model, params, tokenizer, topn: int = 25, temp: float = 0.8,
                 max_model_length: int | None = None, data_sharding=None):
        def _model_apply(params, x, cache=None):
            return model.apply(params, x, cache, generate=True)

        def _generate_loop(i, state):
            tokens, cache, mask, key = state
            key, subkey = jax.random.split(key)
            logits, cache = model.apply(params, tokens, cache, generate=True)
            tokens = sample_top_n(logits + mask, subkey, topn, temp)
            cache = trim_cache(cache)
            return (tokens, cache, mask, key)

        self.topn = topn
        self.temp = temp
        self.params = params
        self.tokenizer = tokenizer
        self.is_modular = hasattr(tokenizer, "subtokenizers")
        self.model_apply = jax.tree_util.Partial(_model_apply)
        self.generate_loop = jax.tree_util.Partial(_generate_loop)
        self.max_model_length = max_model_length
        self.data_sharding = data_sharding

    def _logits_mask(self, subtokenizer_id: str | None) -> Array:
        """Additive (1, 1, vocab) mask restricting sampling to a subtokenizer."""
        if not self.is_modular or subtokenizer_id is None:
            return jnp.zeros((1, 1, self.tokenizer.vocab_size()), dtype=jnp.float32)
        mask_bool = np.asarray(self.tokenizer.mask(subtokenizer_id))
        mask = np.full((1, 1, len(mask_bool)), -np.inf, dtype=np.float32)
        mask[0, 0, mask_bool] = 0.0
        return jnp.asarray(mask)

    def _encode(self, prompt: str, subtokenizer_id: str | None) -> list[int]:
        if self.is_modular:
            assert subtokenizer_id is not None, "a modular tokenizer needs subtokenizer_id"
            return list(self.tokenizer.encode(prompt, subtokenizer_id, bos=True, eos=False))
        return list(self.tokenizer.encode(prompt, bos=True, eos=False))

    def _decode(self, tokens: list[int], subtokenizer_id: str | None) -> str:
        if self.is_modular:
            return self.tokenizer.decode(tokens, subtokenizer_id, strip_bos=True, strip_eos=True)
        return self.tokenizer.decode(tokens, strip_bos=True, strip_eos=True)

    def generate_fast(self, x: Array, key: Array, length: int, mask: Array) -> Array:
        logits, cache = self.model_apply(self.params, x)
        key, subkey = jax.random.split(key)
        tokens = sample_top_n(logits + mask, subkey, self.topn, self.temp)[:, -1:]
        x, cache, _, _ = jax.lax.fori_loop(
            0, length - 1, self.generate_loop, (tokens, cache, mask, key)
        )
        return cache["tokens"]

    def generate(self,
                prompts: list[str] | None = None,
                subtokenizer_id: str | None = None,
                subtokenizer_id_out: str | None = None,
                length: int = 16,
                key: Array | None = None,
                only_return_generated: bool = False,
                token_prompts: list[list[int]] | None = None) -> list[str]:
        """``token_prompts`` bypasses prompt encoding for prompts built directly
        in token space (e.g. few-shot examples mixing two subtokenizers)."""
        key = jax.random.PRNGKey(1234) if key is None else key
        subtokenizer_id_out = subtokenizer_id_out or subtokenizer_id

        assert (prompts is None) != (token_prompts is None), \
            "provide either prompts or token_prompts"
        tokens = token_prompts if prompts is None else \
            [self._encode(p, subtokenizer_id) for p in prompts]
        n_prompts = len(tokens)
        total_length = max(len(t) for t in tokens) + length
        total_length = math.ceil(total_length / 64) * 64
        if self.max_model_length is not None:
            total_length = min(total_length, self.max_model_length)
        tokens = [pad_left(t, self.tokenizer.pad_id(), total_length) for t in tokens]
        tokens = jnp.asarray(tokens)
        if self.data_sharding is not None:
            tokens = jax.device_put(tokens, self.data_sharding)

        mask = self._logits_mask(subtokenizer_id_out)
        generations = self.generate_fast(tokens, key, length, mask)

        # decode prompt and generated tail separately: their token ids belong to
        # different subtokenizers when subtokenizer_id_out != subtokenizer_id
        pad_id = self.tokenizer.pad_id()
        outputs = []
        for i in range(n_prompts):
            ids = generations[i, :].tolist()
            generated = [t for t in ids[-(length - 1):] if t != pad_id]
            text = self._decode(generated, subtokenizer_id_out)
            if not only_return_generated:
                prompt_ids = [t for t in ids[:-(length - 1)] if t != pad_id]
                text = self._decode(prompt_ids, subtokenizer_id) + " " + text
            outputs.append(text)
        return outputs


class GenerateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_dir: str | None = None
    step: int = -1
    prompts: str | None = None            # text file, one prompt per line; None = interactive
    subtokenizer_id: str | None = None    # e.g. "en" or "en,fr"; required for modular tokenizers
    subtokenizer_id_out: str | None = None  # restrict generation to this subtokenizer's vocabulary (default: subtokenizer_id)
    length: int = 64
    topn: int = 25
    temp: float = 0.8
    seed: int = 1234
    only_return_generated: bool = False


def main():
    args = parse_args_to_pydantic_model(GenerateArgs)
    assert args.run_dir, "run_dir is required"
    model, params, tokenizer, step = load_model(args.run_dir, args.step)
    generator = Generator(model, params, tokenizer, topn=args.topn, temp=args.temp)
    key = jax.random.PRNGKey(args.seed)

    if args.prompts:
        with open(args.prompts) as f:
            prompts = [line.rstrip("\n") for line in f if line.strip()]
        for prompt in prompts:
            key, subkey = jax.random.split(key)
            generation = generator.generate(
                [prompt], args.subtokenizer_id, args.subtokenizer_id_out,
                length=args.length, key=subkey,
                only_return_generated=args.only_return_generated,
            )[0]
            print(json.dumps({"prompt": prompt, "generation": generation},
                             ensure_ascii=False), flush=True)
    else:
        while True:
            prompt = input("prompt > ")
            if not prompt:
                break
            t0 = timer()
            key, subkey = jax.random.split(key)
            generation = generator.generate(
                [prompt], args.subtokenizer_id, args.subtokenizer_id_out,
                length=args.length, key=subkey,
            )[0]
            dt = timer() - t0
            print(generation)
            print(f"({dt:.2f} sec, {args.length / dt:.1f} tok/sec)")


if __name__ == "__main__":
    main()
