"""Checkpointing: training save/resume, and model loading for evaluation.

Format (one directory per run):

    run_dir/
    ├── args.json                        resolved config, written once, never overwritten
    ├── <step>/                          orbax composite: params, opt_state, meta
    │                                    (meta: step, world_size, python version)
    └── data_state/<step>/rank_<r>.json  per-rank dataloader snapshot (~1KB of JSON)

Bit-exact resume needs the same world size and the same Python version as the
saved run (the dataloader consumes seeds with ``random.Random(seed).choices``,
whose seed->output mapping Python only guarantees within a version) — the world
size is enforced, a Python mismatch only warns.

``load_model`` also reads the two layouts of checkpoints trained with the
original paper code (a ``<step>/default/`` pytree, or composite items including
``params_avg``). In every layout only ``params`` is restored: the legacy
``params_avg`` running average is never read from disk.
"""

import json
import os
import platform
import sys

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as orbax

from modular_lm.nn.transformer import Transformer, TransformerArgs, build_flash_attention
from modular_tokenizers import TokenizerArgs


def _data_state_dir(run_dir: str, step: int) -> str:
    return os.path.join(run_dir, "data_state", str(step))


def _data_state_path(run_dir: str, step: int, rank: int) -> str:
    return os.path.join(_data_state_dir(run_dir, step), f"rank_{rank}.json")


def checkpoint_steps(run_dir: str) -> list[int]:
    """Steps present in a run directory (any layout), sorted."""
    if not os.path.isdir(run_dir):
        return []
    return sorted(int(d) for d in os.listdir(run_dir) if d.isdigit())


def create_checkpoint_manager(run_dir: str, args=None, rank: int = 0,
                              max_to_keep: int | None = 1, keep_period: int | None = None):
    """Create the run directory, write args.json once (rank 0, never
    overwritten), and return the orbax manager handling step retention."""
    if rank == 0:
        os.makedirs(run_dir, exist_ok=True)
        args_path = os.path.join(run_dir, "args.json")
        if args is not None and not os.path.exists(args_path):
            with open(args_path, "w") as f:
                json.dump(args.model_dump(), f, indent=2, ensure_ascii=False)
    options = orbax.CheckpointManagerOptions(
        max_to_keep=max_to_keep, keep_period=keep_period
    )
    return orbax.CheckpointManager(
        run_dir, options=options, item_names=("params", "opt_state", "meta")
    )


def save_checkpoint(manager, run_dir: str, step: int, params, opt_state,
                    data_state: dict, rank: int):
    meta = {
        "step": step,
        "world_size": jax.process_count(),
        "python_version": platform.python_version(),
    }
    manager.save(
        step,
        args=orbax.args.Composite(
            params=orbax.args.StandardSave(params),
            opt_state=orbax.args.StandardSave(opt_state),
            meta=orbax.args.JsonSave(meta),
        ),
    )
    manager.wait_until_finished()

    # per-rank dataloader snapshot, in a sidecar tree pruned in lock-step with
    # orbax retention
    os.makedirs(_data_state_dir(run_dir, step), exist_ok=True)  # exist_ok: ranks race
    with open(_data_state_path(run_dir, step, rank), "w") as f:
        json.dump(data_state, f)
    if rank == 0:
        kept = set(manager.all_steps()) | {step}
        root = os.path.join(run_dir, "data_state")
        for name in os.listdir(root):
            if name.isdigit() and int(name) not in kept:
                import shutil
                shutil.rmtree(os.path.join(root, name), ignore_errors=True)


def _with_sharding(shapes, shape_to_sharding, dtype=None):
    return jax.tree_util.tree_map(
        lambda s: jax.ShapeDtypeStruct(s.shape, dtype or s.dtype,
                                       sharding=shape_to_sharding(s)),
        shapes,
    )


def restore_checkpoint(run_dir: str, params_shape, optim_shape, shape_to_sharding,
                       rank: int, step: int = -1):
    """Restore full training state for resume. Returns
    ``(params, opt_state, step, data_state)``."""
    step = checkpoint_steps(run_dir)[-1] if step < 0 else step
    handler = orbax.CompositeCheckpointHandler("params", "opt_state", "meta")
    restored = orbax.Checkpointer(handler).restore(
        os.path.join(run_dir, str(step)),
        args=orbax.args.Composite(
            params=orbax.args.StandardRestore(_with_sharding(params_shape, shape_to_sharding)),
            opt_state=orbax.args.StandardRestore(_with_sharding(optim_shape, shape_to_sharding)),
            meta=orbax.args.JsonRestore(),
        ),
    )
    meta = restored["meta"]
    assert meta["world_size"] == jax.process_count(), (
        f"resume world_size mismatch: checkpoint={meta['world_size']} "
        f"current={jax.process_count()} (the data sharding depends on it)"
    )
    if meta["python_version"] != platform.python_version():
        sys.stderr.write(
            f"WARNING: checkpoint was saved with Python {meta['python_version']}, "
            f"resuming with {platform.python_version()}: the resumed data stream "
            "is not guaranteed to be bit-exact.\n"
        )

    state_path = _data_state_path(run_dir, step, rank)
    assert os.path.exists(state_path), (
        f"no dataloader snapshot at {state_path}; cannot resume the data stream"
    )
    with open(state_path) as f:
        data_state = json.load(f)
    return restored["params"], restored["opt_state"], step, data_state


def discard_steps_after(manager, run_dir: str, step: int, rank: int):
    """Delete the checkpoints beyond ``step`` — resuming from an earlier step
    discards the run's later history (orbax would otherwise silently skip
    saving when training reaches a step that still exists)."""
    later_steps = [s for s in manager.all_steps() if s > step]
    for s in later_steps:
        manager.delete(s)
    if rank == 0:
        import shutil
        for s in later_steps:
            shutil.rmtree(_data_state_dir(run_dir, s), ignore_errors=True)
    return later_steps


# ---------------------------------------------------------------------------
# Loading a trained model (evaluation / inference), any layout
# ---------------------------------------------------------------------------

def _model_args_from_checkpoint(saved: dict) -> TransformerArgs:
    """Translate a checkpoint's model config (possibly written by the original
    paper code) to TransformerArgs, dropping the removed options."""
    saved = dict(saved)
    assert saved.pop("use_rope", True), "the released model is RoPE-only"
    assert not saved.pop("qk_norm", False), "qk_norm models are not supported"
    assert not saved.pop("select_index_decoding", False), (
        "select_index_decoding models are not supported"
    )
    saved.pop("max_decoding", None)
    saved.pop("context_size", None)
    # legacy dumps store BOTH the resolved n_heads and head_dim; when head_dim
    # is meaningful (> 0), n_heads was derived from it — re-derive and verify.
    saved_n_heads = saved.get("n_heads")
    if saved.get("head_dim", -1) > 0 and saved_n_heads is not None:
        saved["n_heads"] = None
    args = TransformerArgs(**saved)
    assert saved_n_heads is None or args.n_heads == saved_n_heads
    return args


def _tokenizer_args_from_checkpoint(saved: dict) -> TokenizerArgs:
    saved = dict(saved)
    saved.pop("remove_unused_tokens", None)
    saved.pop("use_hf", None)
    if saved.get("type") == "hf_bpe_modular":
        saved["type"] = "modular"
    # evaluation builds the subtokenizers it needs on demand; the training-time
    # sampled set (possibly np.random-dependent) is not reconstructed.
    saved["sampling_strategy"] = None
    return TokenizerArgs(**saved)


def load_model(run_dir: str, step: int = -1, dtype=jnp.bfloat16,
               tokenizer_args: TokenizerArgs | None = None,
               flash_attention: bool = False, mesh=None, partition_spec=None,
               shape_to_sharding=None):
    """Build the model from a run's args.json and restore its params ONLY.

    ``tokenizer_args`` overrides the checkpoint's tokenizer config (e.g. when
    the tokenizer directory moved since training). Returns
    ``(model, params, tokenizer, step)``.
    """
    with open(os.path.join(run_dir, "args.json")) as f:
        saved_args = json.load(f)

    model_args = _model_args_from_checkpoint(saved_args["model"])
    if tokenizer_args is None:
        tokenizer_args = _tokenizer_args_from_checkpoint(saved_args["tokenizer"])

    tokenizer = tokenizer_args.build_tokenizer()
    model_dict = model_args.model_dump()
    model_dict["vocab_size"] = tokenizer.vocab_size()
    # training packs sequences (no padding, pad_id=-1); inference left-pads
    # prompts, so the padding mask must know the tokenizer's real pad id
    model_dict["pad_id"] = tokenizer.pad_id()
    if flash_attention and mesh is None:
        devices = np.reshape(jax.devices(), (-1, jax.local_device_count()))
        mesh = jax.sharding.Mesh(devices, ("nodes", "gpus"))
        partition_spec = jax.sharding.PartitionSpec(("nodes", "gpus"), None)
    model_dict["flash_attention"] = (
        build_flash_attention(model_args, mesh, partition_spec)
        if flash_attention else None
    )
    model = Transformer(**model_dict)

    def init_params():
        x = jnp.ones((1, 8), dtype="int32")  # param shapes don't depend on batch/context
        return model.init(jax.random.PRNGKey(0), x)

    params_shape = jax.eval_shape(init_params)
    if shape_to_sharding is None:
        def shape_to_sharding(s):
            return jax.sharding.SingleDeviceSharding(jax.devices()[0])
    target = _with_sharding(params_shape, shape_to_sharding, dtype=dtype)

    step = checkpoint_steps(run_dir)[-1] if step < 0 else step
    step_dir = os.path.join(run_dir, str(step))

    if os.path.isdir(os.path.join(step_dir, "meta")) or os.path.isdir(
            os.path.join(step_dir, "params")):
        # new format, or legacy composite items — restore the params item only
        restored = orbax.Checkpointer(orbax.CompositeCheckpointHandler("params")).restore(
            step_dir, args=orbax.args.Composite(params=orbax.args.StandardRestore(target))
        )
        params = restored["params"]
    else:
        # legacy single-pytree layout: {step}/default/ holding
        # {params, params_avg, opt_state}; a partial target reads params only
        restore_args = jax.tree_util.tree_map(
            lambda s: orbax.ArrayRestoreArgs(
                restore_type=jax.Array, dtype=s.dtype, sharding=s.sharding
            ),
            {"params": target},
        )
        params = orbax.Checkpointer(orbax.PyTreeCheckpointHandler()).restore(
            os.path.join(step_dir, "default"),
            args=orbax.args.PyTreeRestore(
                {"params": target}, restore_args=restore_args, transforms={}
            ),
        )["params"]

    return model, params, tokenizer, step
