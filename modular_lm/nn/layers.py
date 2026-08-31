"""Transformer building blocks: MLP, Attention, RMSNorm, Block.

The class names and the order in which submodules are created define the
parameter-tree keys of the checkpoints — do not rename or reorder them.
"""

import math
from collections.abc import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from modular_lm.nn.rope import apply_rope

dkwargs = {"use_bias": False, "dtype": jnp.bfloat16, "precision": "bfloat16"}

GATED_ACTIVATIONS = {"swiglu", "geglu"}


def dense_init(depth: int) -> Callable:
    return nn.initializers.variance_scaling(
        1.0 / (1 + depth), "fan_in", "truncated_normal"
    )


class MLP(nn.Module):
    hidden_dim: int
    depth: int
    activation: str = "swiglu"

    def activation_fn(self, x: Float[Array, "B S F"]) -> Float[Array, "B S F"]:
        x_f32 = x.astype("float32")
        if self.activation == "swiglu":
            return nn.activation.silu(x_f32).astype(x.dtype)
        elif self.activation == "geglu":
            return nn.activation.gelu(x_f32).astype(x.dtype)
        elif self.activation == "gelu":
            return nn.activation.gelu(x_f32).astype(x.dtype)
        elif self.activation == "relu":
            return nn.activation.relu(x_f32).astype(x.dtype)
        elif self.activation == "relu^2":
            return jnp.power(nn.activation.relu(x_f32), 2).astype(x.dtype)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    @nn.compact
    def __call__(self, x: Float[Array, "B S F"]) -> Float[Array, "B S F"]:
        _, _, dim = x.shape
        wh = nn.Dense(self.hidden_dim, kernel_init=dense_init(self.depth), **dkwargs)
        if self.activation in GATED_ACTIVATIONS:
            wg = nn.Dense(
                self.hidden_dim, kernel_init=dense_init(self.depth), **dkwargs
            )
        wo = nn.Dense(dim, kernel_init=dense_init(self.depth), **dkwargs)
        if self.activation in GATED_ACTIVATIONS:
            return wo(wh(x) * self.activation_fn(wg(x)))
        else:
            return wo(self.activation_fn(wh(x)))


class Attention(nn.Module):
    n_heads: int
    depth: int
    flash_attention: Callable | bool | None = None
    gqa: int = 1
    head_dim: int = -1

    @nn.compact
    def __call__(
        self,
        x: Array,
        mask: Array | None = None,
        freqs: Array | None = None,
        kvcache: tuple[Array, Array] | None = None,
    ) -> tuple[Array, tuple[Array, Array]]:
        bsz, _, dim = x.shape
        if self.head_dim > 0:
            dim = self.head_dim * self.n_heads
        wq = nn.Dense(dim, kernel_init=dense_init(self.depth), **dkwargs)
        wk = nn.Dense(dim // self.gqa, kernel_init=dense_init(self.depth), **dkwargs)
        wv = nn.Dense(dim // self.gqa, kernel_init=dense_init(self.depth), **dkwargs)
        wo = nn.Dense(x.shape[-1], kernel_init=dense_init(self.depth), **dkwargs)

        queries = jnp.reshape(wq(x), (bsz, -1, self.n_heads, dim // self.n_heads))
        keys = jnp.reshape(
            wk(x), (bsz, -1, self.n_heads // self.gqa, dim // self.n_heads)
        )
        values = jnp.reshape(
            wv(x), (bsz, -1, self.n_heads // self.gqa, dim // self.n_heads)
        )

        if kvcache is not None:
            keys = jnp.concatenate([kvcache[0], keys], axis=1)
            values = jnp.concatenate([kvcache[1], values], axis=1)
        kvcache = (keys, values)

        if freqs is not None:
            queries, keys = apply_rope(queries, keys, freqs)

        sm_scale = 1.0 / math.sqrt(dim // self.n_heads)
        if callable(self.flash_attention) and x.dtype == jax.dtypes.bfloat16:
            output = self.flash_attention(queries, keys, values)
        else:
            if self.gqa > 1:
                keys = jnp.repeat(keys, self.gqa, axis=2)
                values = jnp.repeat(values, self.gqa, axis=2)
            attn = jnp.einsum("bqhd,bkhd->bhqk", queries, keys, precision="high")
            attn = sm_scale * attn
            if mask is not None:
                attn = attn + mask
            attn = jax.nn.softmax(attn.astype("float32")).astype(x.dtype)
            output = jnp.einsum("bhqk,bkhd->bqhd", attn, values)

        output = jnp.reshape(output, (bsz, -1, dim))
        return wo(output), kvcache


class RMSNorm(nn.Module):
    def _norm(
        self, x: Float[Array, "B S F"], eps: float = 1e-8
    ) -> Float[Array, "B S F"]:
        return x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps)

    @nn.compact
    def __call__(self, x: Float[Array, "B S F"]) -> Float[Array, "B S F"]:
        a = self.param("a", jax.nn.initializers.constant(1.0), (x.shape[-1])).reshape(
            1, 1, -1
        )
        return (a * self._norm(x.astype("float32"))).astype(x.dtype)


class Block(nn.Module):
    mlp_dim: int
    n_heads: int
    depth: int
    flash_attention: Callable | bool | None = None
    activation: str = "swiglu"
    gqa: int = 1
    head_dim: int = -1

    @nn.compact
    def __call__(
        self,
        x: Array,
        mask: Array | None = None,
        freqs: Array | None = None,
        kvcache: tuple[Array, Array] | None = None,
    ) -> tuple[Array, tuple[Array, Array]]:
        attention = Attention(
            self.n_heads,
            self.depth,
            self.flash_attention,
            self.gqa,
            self.head_dim,
        )
        mlp = MLP(self.mlp_dim, self.depth, self.activation)
        norm_1 = RMSNorm()
        norm_2 = RMSNorm()

        h, kvcache = attention(norm_1(x), mask, freqs, kvcache)
        h = x + h
        return h + mlp(norm_2(h)), kvcache
