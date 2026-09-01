"""Summarization evaluation (ROUGE) with few-shot prompting.

Data: a directory holding ``train.jsonl`` and ``<split>.jsonl`` in the
dataset's original format — ``task`` selects the field mapping (xlsum,
eur_lex_sum, wiki_lingua). Sources and summaries are truncated to token
budgets with the evaluating subtokenizer, so prompts fit the context.

Run as ``python -m modular_lm.evaluation.summarization run_dir=... task=xlsum
data_path=... subtokenizer_ids=en length=64 source_max_length=512``.
"""

import json
import os
from itertools import islice

import jax
from pydantic import BaseModel, ConfigDict
from rouge_score import rouge_scorer

from modular_lm.evaluation.utils import batched, report_metrics
from modular_lm.inference.generate import Generator
from modular_lm.training.checkpoint import load_model
from modular_lm.utils import Logger
from modular_tokenizers.config import as_str_list, parse_args_to_pydantic_model

ROUGE_KEYS = ["rouge1", "rouge2", "rougeL", "rougeLsum"]


def load_xlsum(filename, add_title=False):
    with open(filename) as fin:
        for line in fin:
            data = json.loads(line)
            source = data["title"] + "\n" + data["text"] if add_title else data["text"]
            yield {"source": source, "summary": data["summary"]}


def load_eur_lex_sum(filename, add_title=False):
    with open(filename) as fin:
        for line in fin:
            data = json.loads(line)
            yield {"source": data["reference"], "summary": data["summary"]}


def load_wiki_lingua(filename, add_title=False):
    with open(filename) as fin:
        for line in fin:
            data = json.loads(line)
            yield {"source": data["source"], "summary": data["target"]}


LOADERS = {"xlsum": load_xlsum, "eur_lex_sum": load_eur_lex_sum,
           "wiki_lingua": load_wiki_lingua}


class SummarizationArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_dir: str | None = None
    step: int = -1
    task: str | None = None                # xlsum | eur_lex_sum | wiki_lingua
    data_path: str | None = None           # directory with train.jsonl + <split>.jsonl
    split: str = "test"
    add_title: bool = False                # xlsum: prepend the article title

    subtokenizer_ids: str | None = None    # comma-separated; None = full vocabulary
    subtokenizer_id_out: str | None = None
    k_shots: int = 5
    batch_size: int = 16
    source_max_length: int | None = 512    # token budget of each source text
    length: int = 64                       # generated (and shot-summary) tokens
    prefix_q: str = "Text"
    prefix_a: str = "Summary"
    max_examples: int = -1
    flash_attention: bool = False
    output: str | None = None              # append metrics as JSON lines


class TokenTruncater:
    """Truncate text to a token budget under a given subtokenizer."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.is_modular = hasattr(tokenizer, "subtokenizers")

    def __call__(self, text: str, budget: int | None, subtokenizer_id: str | None) -> str:
        if budget is None:
            return text
        if self.is_modular:
            tokens = self.tokenizer.encode(text, subtokenizer_id, bos=False, eos=False)
            return self.tokenizer.decode(list(tokens)[:budget], subtokenizer_id)
        tokens = self.tokenizer.encode(text, bos=False, eos=False)
        return self.tokenizer.decode(list(tokens)[:budget])


def evaluate(generator, tokenizer, args, subtokenizer_id, logger):
    load = LOADERS[args.task]
    truncate = TokenTruncater(tokenizer)

    def format_example(example, with_summary):
        source = truncate(example["source"], args.source_max_length, subtokenizer_id)
        summary = truncate(example["summary"], args.length, subtokenizer_id) if with_summary else ""
        return f"{args.prefix_q}: {source}\n{args.prefix_a}: {summary}".rstrip()

    shots = islice(load(os.path.join(args.data_path, "train.jsonl"), args.add_title),
                   args.k_shots)
    prompt_prefix = "\n\n".join(format_example(e, with_summary=True) for e in shots)

    dataset = load(os.path.join(args.data_path, f"{args.split}.jsonl"), args.add_title)
    if args.max_examples > 0:
        dataset = islice(dataset, args.max_examples)

    def build_prompt(example) -> str:
        return prompt_prefix + "\n\n" + format_example(example, with_summary=False)

    scorer = rouge_scorer.RougeScorer(ROUGE_KEYS, use_stemmer=True)
    rouge_sums = {key: 0.0 for key in ROUGE_KEYS}
    n_examples = 0
    key = jax.random.PRNGKey(1234)

    for batch in batched(dataset, args.batch_size):
        key, subkey = jax.random.split(key)
        outputs = generator.generate(
            [build_prompt(example) for example in batch],
            subtokenizer_id=subtokenizer_id,
            subtokenizer_id_out=args.subtokenizer_id_out,
            length=args.length, key=subkey, only_return_generated=True)

        for example, output in zip(batch, outputs):
            prediction = output.split("\n\n")[0].strip()
            rouge = scorer.score(example["summary"], prediction)
            for rouge_key in ROUGE_KEYS:
                rouge_sums[rouge_key] += rouge[rouge_key].fmeasure
            n_examples += 1

        logger.log(n_examples=n_examples,
                   **{k: round(v / n_examples, 4) for k, v in rouge_sums.items()})

    return {key: v / n_examples for key, v in rouge_sums.items()} | {"n_examples": n_examples}


def main():
    args = parse_args_to_pydantic_model(SummarizationArgs)
    assert args.run_dir and args.task and args.data_path, \
        "run_dir, task and data_path are required"
    assert args.task in LOADERS, f"unknown task: {args.task} (choose from {sorted(LOADERS)})"
    logger = Logger()
    logger.log(json.dumps(args.model_dump()))

    model, params, tokenizer, step = load_model(
        args.run_dir, args.step, flash_attention=args.flash_attention
    )
    generator = Generator(model, params, tokenizer, topn=1)
    subtokenizer_ids = as_str_list(args.subtokenizer_ids) if args.subtokenizer_ids else [None]

    for subtokenizer_id in subtokenizer_ids:
        metrics = evaluate(generator, tokenizer, args, subtokenizer_id, logger)
        run_info = dict(task=args.task, split=args.split, k_shots=args.k_shots,
                        step=step, subtokenizer_id=subtokenizer_id,
                        subtokenizer_id_out=args.subtokenizer_id_out,
                        run_dir=args.run_dir, data_path=args.data_path, length=args.length)
        report_metrics(metrics, run_info, logger, args.output)


if __name__ == "__main__":
    main()
