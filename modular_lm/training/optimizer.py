"""Optimizer configuration: AdamW with gradient clipping, and the learning-rate
schedules used by the paper (warmup-cosine) and its extensions (warmup-stable-decay).
"""

import jax.numpy as jnp
import optax
from pydantic import BaseModel, ConfigDict


def wsd_schedule(lr, warmup_steps, decay_start=None, decay_steps=None,
                 final_lr_frac=0.0, decay_type="linear", total_steps=None):
    """Warmup-Stable-Decay.

    warmup: 0 -> lr over `warmup_steps`
    stable: constant `lr` (flat) -- so training can be EXTENDED arbitrarily; the
            stable LR doesn't depend on total_steps.
    decay:  lr -> final_lr_frac*lr over `decay_steps`, starting at `decay_start`.

    Open-ended training: leave `decay_start=None` to stay flat forever (just keep
    raising optim.total_steps to train longer). When you want a final model, resume
    with decay_start=<current step> and decay_steps=<D> to anneal the tail.
    """
    warmup = optax.linear_schedule(0.0, lr, warmup_steps)

    if decay_start is None:
        # warmup then flat forever (no annealing yet)
        return optax.join_schedules(
            [warmup, optax.constant_schedule(lr)], [warmup_steps]
        )

    if decay_steps is None:
        decay_steps = (total_steps - decay_start) if total_steps else max(1, decay_start // 10)
    assert decay_start >= warmup_steps, "decay_start must be >= warmup_steps"
    assert decay_steps > 0, "decay_steps must be > 0"
    final_lr = final_lr_frac * lr

    if decay_type == "linear":
        decay = optax.linear_schedule(lr, final_lr, decay_steps)
    elif decay_type == "cosine":
        decay = optax.cosine_decay_schedule(lr, decay_steps, final_lr_frac)
    elif decay_type == "1-sqrt":
        # MiniCPM-style: lr * (1 - sqrt(t / decay_steps))
        def decay(step):
            t = jnp.clip(step / decay_steps, 0.0, 1.0)
            return lr - (lr - final_lr) * jnp.sqrt(t)
    else:
        raise ValueError(f"unknown decay_type: {decay_type}")

    return optax.join_schedules(
        [warmup, optax.constant_schedule(lr), decay],
        [warmup_steps, decay_start],
    )


class OptimArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lr: float = 5e-4
    lr_coef: float = 0.1             # final LR as a fraction of lr
    warmup_steps: int = 500
    total_steps: int = 50000
    scheduler: str = "cosine"        # cosine | wsd

    # WSD (warmup-stable-decay) controls
    decay_start: int | None = None   # step to begin annealing; None => flat stable forever
    decay_steps: int | None = None
    decay_type: str = "linear"       # linear | cosine | 1-sqrt

    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    eps: float = 1e-8
    clip: float = 1.0

    def build_scheduler(self):
        if self.scheduler == "cosine":
            return optax.warmup_cosine_decay_schedule(
                0.0, self.lr, self.warmup_steps, self.total_steps, self.lr_coef * self.lr,
            )
        if self.scheduler == "wsd":
            return wsd_schedule(
                self.lr, self.warmup_steps,
                decay_start=self.decay_start,
                decay_steps=self.decay_steps,
                final_lr_frac=self.lr_coef,
                decay_type=self.decay_type,
                total_steps=self.total_steps,
            )
        raise ValueError(f"unknown scheduler: {self.scheduler}")

    def build_optim(self):
        return optax.chain(
            optax.clip_by_global_norm(self.clip),
            optax.adamw(
                self.build_scheduler(),
                b1=self.beta1,
                b2=self.beta2,
                weight_decay=self.weight_decay,
                eps=self.eps,
            ),
        )
