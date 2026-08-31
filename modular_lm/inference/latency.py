"""Forward-pass latency of a model as a function of its vocabulary size.

The paper's efficiency measurement: a subtokenizer's compact vocabulary
shrinks the embedding and output layers, so the same architecture runs faster
than with the full merged vocabulary. Parameters are randomly initialized —
this measures the architecture's compute cost, not model quality — while the
token streams come from real text so sequence statistics are realistic.

Run as ``python -m modular_lm.inference.latency data_path=... tokenizer.path=...
subtokenizer_id=en context_sizes=1024,2048`` — measurements are appended as
JSON lines to ``output`` when given.
"""

import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from modular_lm.data.utils import iter_file_lines, source_files
from modular_lm.nn.transformer import Transformer, TransformerArgs
from modular_lm.utils import Logger, format_number, params_count
from modular_tokenizers import TokenizerArgs
from modular_tokenizers.config import as_str_list, parse_args_to_pydantic_model


class LatencyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data_path: str | None = None      # text corpus: .txt/.jsonl/.zst file or directory
    text_field: str = "text"
    n_docs: int = 1000
    context_sizes: str = "2048"       # comma-separated, one measurement each
    batch_size: int = 1

    @field_validator("context_sizes", mode="before")
    @classmethod
    def _as_str(cls, value):
        if isinstance(value, (list, tuple)):
            return ",".join(str(v) for v in value)
        return str(value)
    n_repeat: int = 5
    jit: bool = False

    subtokenizer_id: str | None = None  # measure this subtokenizer's compact vocab
    # tokenize with the subtokenizer but size the model at the FULL merged
    # vocabulary (the cost a non-modular model pays on the same token stream)
    use_global_vocab: bool = False

    output: str | None = None           # append measurements as JSON lines
    tokenizer: TokenizerArgs = TokenizerArgs()
    model: TransformerArgs = TransformerArgs()


def iter_docs(path: str, text_field: str):
    for file in source_files(path):
        for line in iter_file_lines(file):
            if file.endswith(".txt"):
                if line.strip():
                    yield line.rstrip("\n")
            else:
                yield json.loads(line)[text_field]


def tokenize_batches(tokenizer, args: LatencyArgs, context_size: int) -> np.ndarray:
    tokens = []
    for n, text in enumerate(iter_docs(args.data_path, args.text_field)):
        if n == args.n_docs:
            break
        tokens.extend(tokenizer.encode(text, bos=False, eos=False))
    n_per_batch = args.batch_size * context_size
    n_batches = max(len(tokens) // n_per_batch, 1)
    tokens = tokens[: n_batches * n_per_batch]
    tokens += [0] * (n_batches * n_per_batch - len(tokens))
    return np.array(tokens).reshape(n_batches, args.batch_size, context_size)


def main():
    args = parse_args_to_pydantic_model(LatencyArgs)
    assert args.data_path, "data_path is required"
    logger = Logger()

    tokenizer = args.tokenizer.build_tokenizer()
    global_vocab_size = tokenizer.vocab_size()
    if args.subtokenizer_id is not None:
        tokenizer = tokenizer.build_subtokenizer(args.subtokenizer_id)
    vocab_size = global_vocab_size if args.use_global_vocab else tokenizer.vocab_size()
    logger.log(subtokenizer=args.subtokenizer_id, vocab_size=vocab_size,
               global_vocab_size=global_vocab_size)

    for context_size in as_str_list(args.context_sizes):
        context_size = int(context_size)
        model_dict = args.model.model_dump()
        model_dict["vocab_size"] = vocab_size
        model_dict["flash_attention"] = None
        model = Transformer(**model_dict)
        params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 8), dtype=int))
        logger.log(context_size=context_size,
                   parameters=format_number(params_count(params)))

        def forward_once(params, tokens):
            logits, _ = model.apply(params, tokens)
            return jax.nn.softmax(logits, axis=-1)

        if args.jit:
            forward_once = jax.jit(forward_once)

        def forward(params, batches):
            for i in range(batches.shape[0]):
                probs = forward_once(params, batches[i])
            return probs

        batches = jnp.asarray(tokenize_batches(tokenizer, args, context_size))
        logger.log(batches=batches.shape[0], batch_shape=str(batches.shape[1:]))

        start = time.time()
        forward(params, batches).block_until_ready()
        logger.log(warmup_time=round(time.time() - start, 4))

        times = []
        for i in range(args.n_repeat):
            start = time.time()
            forward(params, batches).block_until_ready()
            times.append(time.time() - start)
            logger.log(**{f"run_{i}": round(times[-1], 4)})
        mean_time = sum(times) / len(times)
        logger.log(mean_execution_time=round(mean_time, 4))

        if args.output:
            record = args.model_dump()
            record.update(
                context_size=context_size,
                vocab_size=vocab_size,
                global_vocab_size=global_vocab_size,
                execution_times=times,
                mean_execution_time=mean_time,
            )
            with open(args.output, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
