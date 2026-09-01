"""Pretraining entry point.

Run as ``python -m modular_lm.train key=value ...`` (OmegaConf dotted
overrides, optionally layered on YAML files via ``config=path.yaml``).

Each training batch is monolingual and tokenized with one sampled subtokenizer
(paper §4); the loss is restricted to that subtokenizer's vocabulary through an
additive logits mask, and scaled to bits-per-byte with the batch's tokens/bytes
ratio. Resume is automatic (and bit-exact for the data stream) when ``run_dir``
already contains a checkpoint.
"""

import itertools
import json
import os
from dataclasses import dataclass, field
from collections import defaultdict
from timeit import default_timer as timer

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import PartitionSpec as P
from pydantic import BaseModel, ConfigDict

from modular_lm.data.pretrain import Batch, PretrainDataConfig, PretrainDataset
from modular_lm.nn.transformer import TransformerArgs, create_model
from modular_lm.training.checkpoint import (
    checkpoint_steps,
    create_checkpoint_manager,
    discard_steps_after,
    restore_checkpoint,
    save_checkpoint,
)
from modular_lm.training.optimizer import OptimArgs
from modular_lm.utils import Logger, format_number, global_array, params_count
from modular_tokenizers import TokenizerArgs
from modular_tokenizers.config import parse_args_to_pydantic_model

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
# The latency-hiding scheduler keeps more buffers live; with full-vocabulary
# logits this costs the few GB that can push a large run out of memory.
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_latency_hiding_scheduler=false"
).strip()


class TrainArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # checkpoints, args.json and metrics.jsonl land here; None disables saving
    run_dir: str | None = None
    resume: bool = False    # continue the run_dir's training (data stream bit-exact)
    resume_step: int = -1   # checkpoint step to resume from; -1 = the latest
    save_freq: int = 5000
    max_ckpt_to_keep: int | None = 1
    keep_period: int | None = 20000   # additionally keep every keep_period-th step
    log_freq: int = 100
    distributed: bool = False
    seed: int = 1234

    model: TransformerArgs = TransformerArgs()
    data: PretrainDataConfig = PretrainDataConfig()
    optim: OptimArgs = OptimArgs()
    tokenizer: TokenizerArgs = TokenizerArgs()


@dataclass
class Metrics:
    loss: float = 0.0
    n_batches: int = 0
    t0: float = 0.0
    loss_per_lang: defaultdict = field(default_factory=lambda: defaultdict(float))
    count_per_lang: defaultdict = field(default_factory=lambda: defaultdict(int))


def main():
    args = parse_args_to_pydantic_model(
        TrainArgs, replace_paths=("data.sources", "data.weights")
    )
    rank = int(os.environ.get("SLURM_PROCID", 0))
    logger = Logger(dummy=rank != 0)

    # Must initialize the distributed runtime BEFORE jax.devices().
    # IMPORTANT: jax's SLURM auto-detect assumes ONE process per GPU and would
    # claim a single GPU per process. With one task per node owning all its
    # GPUs, claim every locally-visible GPU explicitly.
    if args.distributed:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        n_local = len([d for d in cvd.split(",") if d != ""])
        jax.distributed.initialize(
            local_device_ids=list(range(n_local)) if n_local else None
        )

    devices = np.reshape(jax.devices(), (-1, jax.local_device_count()))
    mesh = jax.sharding.Mesh(devices, ("nodes", "gpus"))
    partition_spec = P(("nodes", "gpus"), None)
    logger.log(mesh=str(mesh.devices.shape), axes=str(mesh.axis_names))

    # Data-weighted subtokenizer sets are drawn with np.random at build time:
    # seed BEFORE building so every rank (and every resume) builds the same set.
    np.random.seed(args.seed)
    model, tokenizer = create_model(args, mesh=mesh, partition_spec=partition_spec)
    logger.log(vocab_size=tokenizer.vocab_size())

    if hasattr(tokenizer, "subtokenizer_ids_by_lang"):
        missing = set(args.data.sources) - set(tokenizer.subtokenizer_ids_by_lang)
        assert not missing, f"data languages unknown to the tokenizer: {sorted(missing)}"

    optimizer = args.optim.build_optim()
    schedule = args.optim.build_scheduler()

    def init_params_and_optim():
        x = jnp.ones(
            (jax.process_count() * args.data.batch_size, args.data.context_size)
        ).astype("int32")
        params = model.init(jax.random.PRNGKey(0), x)
        return params, optimizer.init(params)

    def shape_to_sharding(s):
        if len(s.shape) <= 1:
            return jax.sharding.NamedSharding(mesh, P())
        return jax.sharding.NamedSharding(mesh, P(("gpus"), None))

    params_shape, optim_shape = jax.eval_shape(init_params_and_optim)
    params_sharding = jax.tree_util.tree_map(shape_to_sharding, params_shape)
    optim_sharding = jax.tree_util.tree_map(shape_to_sharding, optim_shape)
    none_sharding = jax.sharding.NamedSharding(mesh, P())
    data_sharding = jax.sharding.NamedSharding(mesh, P(("nodes", "gpus"), None))

    manager = None
    if args.run_dir:
        manager = create_checkpoint_manager(
            args.run_dir, args=args, rank=rank,
            max_to_keep=args.max_ckpt_to_keep, keep_period=args.keep_period,
        )

    data_state = None
    if args.resume:
        assert args.run_dir, "resume=true requires run_dir"
        assert checkpoint_steps(args.run_dir), \
            f"resume=true but no checkpoint found in {args.run_dir}"
        params, opt_state, saved_step, data_state = restore_checkpoint(
            args.run_dir, params_shape, optim_shape, shape_to_sharding,
            rank=rank, step=args.resume_step,
        )
        start = saved_step + 1
        logger.log(f"Resumed from {args.run_dir} at step {saved_step}")
        if manager:
            discarded = discard_steps_after(manager, args.run_dir, saved_step, rank)
            if discarded:
                logger.log(f"WARNING: discarded the run's later history "
                           f"(checkpoints at steps {discarded})")
    else:
        assert not (args.run_dir and checkpoint_steps(args.run_dir)), (
            f"{args.run_dir} already contains checkpoints: pass resume=true to "
            "continue it, or use a fresh run_dir"
        )
        start = 0
        params, opt_state = jax.jit(
            init_params_and_optim, out_shardings=(params_sharding, optim_sharding)
        )()

    logger.log(json.dumps(args.model_dump()))
    logger.log(parameters=format_number(params_count(params)))

    dataset = PretrainDataset(
        args.data, tokenizer, rank=jax.process_index(), world_size=jax.process_count()
    )
    if data_state is not None:
        dataset.restore(data_state)
        logger.log("Restored the dataloader state (bit-exact resume)")

    # One-hot cross-entropy on purpose: XLA fuses one_hot->CE into the logits
    # matmul epilogue; the integer-label form breaks that fusion and OOMs at
    # scale. c_bpb converts the loss to bits-per-byte, making it comparable
    # across subtokenizers.
    def loss_fn(params, batch, c_bpb):
        logits, _ = model.apply(params, batch["x"])
        logits = logits + jnp.expand_dims(batch["logits_mask"], axis=1)
        labels = jax.nn.one_hot(batch["y"], logits.shape[-1])
        ce = optax.softmax_cross_entropy(logits, labels)
        return c_bpb * jnp.mean(ce)

    def train_step(params, opt_state, batch, c_bpb):
        value, grads = jax.value_and_grad(loss_fn)(params, batch, c_bpb)
        update, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, update)
        return params, opt_state, value

    batch_shardings = {"x": data_sharding, "y": data_sharding, "logits_mask": data_sharding}
    train_step = jax.jit(
        train_step,
        in_shardings=(params_sharding, optim_sharding, batch_shardings, none_sharding),
        out_shardings=(params_sharding, optim_sharding, none_sharding),
        donate_argnums=(0, 1),
    )

    metrics_file = None
    if args.run_dir and rank == 0:
        metrics_file = open(os.path.join(args.run_dir, "metrics.jsonl"), "a")

    if start > args.optim.total_steps:
        logger.log(f"Run already at step {start - 1} >= total_steps: nothing to do")
        return

    tokens_per_gpu = args.data.batch_size * args.data.context_size // jax.local_device_count()
    metrics = Metrics(t0=timer())
    logger.log("Start training...")

    for i in itertools.count(start):
        batch: Batch = next(dataset)
        device_batch = {
            "x": global_array(batch.x, mesh, data_sharding),
            "y": global_array(batch.y, mesh, data_sharding),
            "logits_mask": global_array(batch.logits_mask, mesh, data_sharding),
        }
        params, opt_state, value = train_step(
            params, opt_state, device_batch, jnp.float32(batch.c_bpb)
        )

        value = value.item()
        metrics.loss += value
        metrics.n_batches += 1
        # attribution of the global loss to this rank's batch language — exact
        # in single-process runs, an approximation across processes
        metrics.loss_per_lang[batch.lang] += value
        metrics.count_per_lang[batch.lang] += 1

        if np.isnan(value):
            logger.log("Houston, we have NaNs!")
            break

        if i % args.log_freq == 0:
            dt = timer() - metrics.t0
            per_lang = {
                lang: round(metrics.loss_per_lang[lang] / metrics.count_per_lang[lang], 4)
                for lang in sorted(metrics.loss_per_lang)
            }
            record = {
                "step": i,
                "loss": round(metrics.loss / metrics.n_batches, 4),
                "lr": float(schedule(i)),
                "wps": int(metrics.n_batches * tokens_per_gpu / dt),
                **{f"loss_{lang}": v for lang, v in per_lang.items()},
            }
            logger.log(**record)
            if metrics_file:
                metrics_file.write(json.dumps(record) + "\n")
                metrics_file.flush()
            metrics = Metrics(t0=timer())

        # the dataloader snapshot is taken at the step boundary (batch i
        # consumed, nothing prefetched), which resume replays from exactly
        if manager and args.save_freq > 0 and i % args.save_freq == 0:
            logger.log("Saving checkpoint...")
            save_checkpoint(manager, args.run_dir, i, params, opt_state,
                            dataset.snapshot(), rank=rank)

        if i >= args.optim.total_steps:
            break


if __name__ == "__main__":
    main()
