"""Shared helpers: logging, distributed synchronization, array placement."""

import datetime

import jax
import jax.numpy as jnp
import numpy as np


def get_date() -> str:
    return datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S")


class Logger:
    """Timestamped key/value logging to stdout; ``dummy=True`` silences a rank."""

    def __init__(self, dummy: bool = False):
        self.dummy = dummy

    def log(self, msg: str = "", **kwargs):
        if self.dummy:
            return
        parts = [get_date(), msg] if msg else [get_date()]
        parts += [f"{k}: {v}" for k, v in kwargs.items()]
        print(" | ".join(parts), flush=True)


def sync():
    # cross-host barrier: pmap leading axis must equal the local device count
    if jax.process_count() == 1:
        return
    jax.pmap(lambda x: jax.lax.psum(x, "i"), axis_name="i")(
        jnp.ones(jax.local_device_count())
    )


def global_array(local_array, mesh, sharding):
    """Assemble each process's local batch rows into one global device array."""
    bsz, n = local_array.shape
    global_shape = (jax.process_count() * bsz, n)
    arrays = jax.device_put(
        jnp.split(local_array, len(mesh.local_devices), axis=0), mesh.local_devices
    )
    return jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)


def params_count(params) -> int:
    return sum(
        e.size for e in jax.tree_util.tree_flatten(params)[0]
        if isinstance(e, (jax.Array, np.ndarray))
    )


def format_number(n) -> str:
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= scale:
            return f"{n / scale:.2f}{suffix}"
    return f"{n}"
