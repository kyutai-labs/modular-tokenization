"""Rotary position embeddings (RoPE), computed with complex numbers.

Adjacent feature dimensions are rotated as pairs: (d0, d1), (d2, d3), ...
This pairing is part of the trained models' geometry and must not change.
"""

import jax.numpy as jnp
from jaxtyping import Array


def to_complex(x: Array) -> Array:
    x = x.reshape(*x.shape[:-1], -1, 2)
    return x[..., 0] + 1.0j * x[..., 1]


def to_real(x: Array) -> Array:
    x = jnp.stack([jnp.real(x), jnp.imag(x)], axis=-1)
    return x.reshape(*x.shape[:-2], -1)


def compute_rope_freqs(dimension: int, length: int, theta: float = 100000.0) -> Array:
    assert dimension % 2 == 0
    t = jnp.arange(length)
    x = jnp.arange(0, dimension, 2)
    freqs = 1.0 / (theta ** (x / dimension))
    freqs = jnp.outer(t, freqs)
    return jnp.exp(1.0j * freqs)


def apply_rope(queries: Array, keys: Array, freqs: Array) -> tuple[Array, Array]:
    n = queries.shape[1]
    queries = to_real(freqs[-n:, ...] * to_complex(queries)).astype(queries.dtype)
    keys = to_real(freqs * to_complex(keys)).astype(keys.dtype)
    return queries, keys
