"""Decoder-only transformer with RoPE, and its configuration.

`create_model` builds the model and its (modular) tokenizer from the run
configuration, wiring flash attention across the device mesh when enabled.
"""

import math
from collections.abc import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array
from pydantic import BaseModel, ConfigDict

from modular_lm.nn.layers import Block, RMSNorm, dense_init, dkwargs
from modular_lm.nn.rope import compute_rope_freqs


def emb_init(key: Array, shape: tuple[int, ...], dtype) -> Array:
    return jax.random.uniform(key, shape, minval=-0.1, maxval=0.1)


class Transformer(nn.Module):
    dim: int
    mlp_dim: int
    n_heads: int
    n_layers: int
    vocab_size: int
    theta_rope: float
    flash_attention: Callable | bool | None = None
    pad_id: int = -1
    activation: str = "swiglu"
    gqa: int = 1
    head_dim: int = -1

    @nn.compact
    def __call__(
        self,
        x: Array,
        cache: dict[str, Array | tuple[Array, Array]] | None = None,
        generate: bool = False,
    ) -> tuple[Array, dict[str, Array]]:
        bsz, n = x.shape
        new_cache = {}

        if cache:
            offset = cache["tokens"].shape[1]
            new_cache["tokens"] = jnp.concatenate([cache["tokens"], x], axis=1)
        else:
            offset = 0
            new_cache["tokens"] = x

        mask_padding = new_cache["tokens"] == self.pad_id
        mask_padding = jnp.expand_dims(mask_padding, 1).repeat(n, axis=1)
        mask_padding = ~jnp.tril(mask_padding, k=offset - 1)

        mask_causal = jnp.expand_dims(jnp.ones((n, n + offset)), 0)
        mask_causal = jnp.tril(mask_causal, k=offset)

        mask = jnp.log(mask_padding * mask_causal)
        mask = jnp.expand_dims(mask, 1)

        h = nn.Embed(
            self.vocab_size,
            self.dim,
            embedding_init=emb_init,
            dtype=jax.dtypes.bfloat16,
        )(x)

        head_dim = self.head_dim if self.head_dim > 0 else self.dim // self.n_heads
        freqs = compute_rope_freqs(head_dim, n + offset, self.theta_rope)
        freqs = jnp.expand_dims(freqs, 1)

        CBlock = nn.checkpoint(Block, policy=jax.checkpoint_policies.nothing_saveable())
        for i in range(self.n_layers):
            block_i = CBlock(
                self.mlp_dim,
                self.n_heads,
                i,
                self.flash_attention,
                self.activation,
                self.gqa,
                self.head_dim,
            )
            cache_i = cache[f"layer_{i}"] if cache else None
            h, cache_i = block_i(h, mask, freqs, cache_i)
            new_cache[f"layer_{i}"] = cache_i

        h = RMSNorm()(h)
        if generate:
            h = h[:, -1:, :]
        h = nn.Dense(self.vocab_size, kernel_init=dense_init(self.n_layers), **dkwargs)(
            h
        )
        return h.astype("float32"), new_cache


class TransformerArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dim: int = 1024
    mlp_dim: int = 2816
    n_heads: int | None = None
    n_layers: int = 24
    vocab_size: int | None = None
    theta_rope: float = 100000.0
    flash_attention: bool = True
    pad_id: int = -1
    activation: str = "swiglu"
    gqa: int = 1
    head_dim: int = 128

    def model_post_init(self, *args):
        assert self.theta_rope > 0, "theta_rope must be > 0"
        if self.n_heads is None:
            assert self.head_dim > 0, "head_dim must be > 0 if n_heads is not set"
            self.n_heads = self.dim // self.head_dim
        else:
            self.head_dim = -1


def build_flash_attention(model_args: TransformerArgs, mesh, partition_spec):
    """Flash-attention kernel call, sharded across the device mesh."""
    from functools import partial

    from flash_attn3_jax import flash_mha

    head_dim = model_args.head_dim
    if head_dim <= 0:
        head_dim = model_args.dim // model_args.n_heads
    return jax.shard_map(
        partial(flash_mha, is_causal=True, softmax_scale=1.0 / math.sqrt(head_dim)),
        mesh=mesh,
        in_specs=(partition_spec, partition_spec, partition_spec),
        out_specs=(partition_spec),
        check_vma=False,
    )


def create_model(args, mesh=None, partition_spec=None):
    flash_attention = None
    if args.model.flash_attention:
        flash_attention = build_flash_attention(args.model, mesh, partition_spec)

    tokenizer = args.tokenizer.build_tokenizer()

    model_args = args.model.model_dump()
    model_args["vocab_size"] = tokenizer.vocab_size()
    model_args["flash_attention"] = flash_attention
    model = Transformer(**model_args)

    return model, tokenizer
