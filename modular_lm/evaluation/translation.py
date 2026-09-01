"""Machine-translation evaluation (BLEU, plus EM/F1) with few-shot prompting.

Data: a directory holding ``{train,valid,test}.jsonl`` of parallel examples
``{"question": <source sentence>, "answer": [<references>...]}`` (FLORES-style).
Two directions are supported from the same files: ``reverse=true`` swaps
question and answer. ``target_data_path`` gives a second, question-aligned
directory to translate between two non-English languages through their shared
English questions.

Cross-subtokenizer mode: when ``subtokenizer_id_out`` differs from
``subtokenizer_id``, the few-shot prompt is built directly in token space —
source sides encoded with the input subtokenizer, target sides with the output
one — and generation is restricted to the output subtokenizer.

Run as ``python -m modular_lm.evaluation.translation run_dir=... data_path=...
subtokenizer_ids=fr subtokenizer_id_out=en target_lang=en length=64``.
"""

import json
import os
from itertools import islice

import jax
from pydantic import BaseModel, ConfigDict
from sacrebleu.metrics import BLEU

from modular_lm.evaluation.utils import batched, exact_match, f1_score, report_metrics
from modular_lm.inference.generate import Generator
from modular_lm.training.checkpoint import load_model
from modular_lm.utils import Logger
from modular_tokenizers.config import as_str_list, parse_args_to_pydantic_model


def load_parallel(filename: str, target_filename: str | None = None, reverse: bool = False):
    """Yield {question, answer} pairs; see the module docstring for the format."""
    if target_filename is None:
        with open(filename) as fin:
            for line in fin:
                data = json.loads(line)
                if reverse:
                    yield {"question": data["answer"][0], "answer": [data["question"]]}
                else:
                    yield {"question": data["question"], "answer": data["answer"]}
    else:
        with open(filename) as fin, open(target_filename) as fin_target:
            for line, line_target in zip(fin, fin_target):
                data, data_target = json.loads(line), json.loads(line_target)
                assert data["question"] == data_target["question"], \
                    "source and target files must be question-aligned"
                yield {"question": data["answer"][0], "answer": data_target["answer"]}


def build_bleu(target_lang: str | None) -> BLEU:
    tokenize = {"zh": "zh", "ja": "ja-mecab"}.get(target_lang, "13a")
    return BLEU(tokenize=tokenize, lowercase=True)


class TranslationArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_dir: str | None = None
    step: int = -1
    data_path: str | None = None          # directory with {train,valid,test}.jsonl
    target_data_path: str | None = None   # optional question-aligned target directory
    reverse: bool = False                 # swap question and answer (opposite direction)
    split: str = "test"
    target_lang: str | None = None        # selects the BLEU tokenizer (zh / ja / other)

    subtokenizer_ids: str | None = None   # comma-separated; None = full vocabulary
    subtokenizer_id_out: str | None = None
    k_shots: int = 5
    batch_size: int = 16
    length: int = 64                      # generated tokens per example
    prefix_q: str = "Q"
    prefix_a: str = "A"
    max_examples: int = -1
    flash_attention: bool = False
    output: str | None = None             # append metrics as JSON lines


def make_prompt_builder(tokenizer, shots, args, subtokenizer_id):
    """Return ``build_prompt(example) -> str | list[int]``: the few-shot prefix
    plus the example's question. In cross-subtokenizer mode the prompt is a
    token list (source sides encoded with the input subtokenizer, target sides
    with the output one); otherwise it is plain text."""

    def question_line(example):
        return f"{args.prefix_q}: {example['question']} {args.prefix_a}:"

    if args.subtokenizer_id_out is None:
        shot_lines = [f"{question_line(e)} {e['answer'][0]}" for e in shots]

        def build_prompt(example) -> str:
            return "\n".join(shot_lines + [question_line(example)])
    else:
        shot_tokens = [tokenizer.bos_id()]
        for e in shots:
            shot_tokens += list(tokenizer.encode(question_line(e), subtokenizer_id,
                                                 bos=False, eos=False))
            shot_tokens += list(tokenizer.encode(f"{e['answer'][0]}\n",
                                                 args.subtokenizer_id_out,
                                                 bos=False, eos=False))

        def build_prompt(example) -> list[int]:
            return shot_tokens + list(tokenizer.encode(question_line(example),
                                                       subtokenizer_id, bos=False, eos=False))

    return build_prompt


def evaluate(generator, tokenizer, args, subtokenizer_id, logger):
    # protocol: shots come from train.jsonl, or valid.jsonl when more than 5
    shots_file = "train.jsonl" if args.k_shots <= 5 else "valid.jsonl"

    def task_files(name):
        target = os.path.join(args.target_data_path, name) if args.target_data_path else None
        return os.path.join(args.data_path, name), target

    shots = list(islice(load_parallel(*task_files(shots_file), reverse=args.reverse),
                        args.k_shots))
    dataset = load_parallel(*task_files(f"{args.split}.jsonl"), reverse=args.reverse)
    if args.max_examples > 0:
        dataset = islice(dataset, args.max_examples)

    build_prompt = make_prompt_builder(tokenizer, shots, args, subtokenizer_id)
    cross_subtokenizer = args.subtokenizer_id_out is not None

    bleu = build_bleu(args.target_lang)
    predictions, references = [], []
    em = f1 = 0.0
    key = jax.random.PRNGKey(1234)

    for batch in batched(dataset, args.batch_size):
        key, subkey = jax.random.split(key)
        prompts = [build_prompt(example) for example in batch]
        if cross_subtokenizer:
            outputs = generator.generate(
                token_prompts=prompts, subtokenizer_id=subtokenizer_id,
                subtokenizer_id_out=args.subtokenizer_id_out,
                length=args.length, key=subkey, only_return_generated=True)
        else:
            outputs = generator.generate(
                prompts, subtokenizer_id=subtokenizer_id,
                length=args.length, key=subkey, only_return_generated=True)

        for example, output in zip(batch, outputs):
            prediction = output.split("\n")[0].strip()
            predictions.append(prediction)
            references.append(example["answer"][0])
            em += exact_match(prediction, example["answer"])
            f1 += f1_score(prediction, example["answer"])

        logger.log(n_examples=len(predictions),
                   bleu=round(bleu.corpus_score(predictions, [references]).score, 2),
                   em=round(100 * em / len(predictions), 1),
                   f1=round(100 * f1 / len(predictions), 1))

    n_examples = len(predictions)
    return {
        "bleu": bleu.corpus_score(predictions, [references]).score,
        "exact_match": 100 * em / n_examples,
        "f1": 100 * f1 / n_examples,
        "n_examples": n_examples,
    }


def main():
    args = parse_args_to_pydantic_model(TranslationArgs)
    assert args.run_dir and args.data_path, "run_dir and data_path are required"
    logger = Logger()
    logger.log(json.dumps(args.model_dump()))

    model, params, tokenizer, step = load_model(
        args.run_dir, args.step, flash_attention=args.flash_attention
    )
    generator = Generator(model, params, tokenizer, topn=1)
    subtokenizer_ids = as_str_list(args.subtokenizer_ids) if args.subtokenizer_ids else [None]

    for subtokenizer_id in subtokenizer_ids:
        metrics = evaluate(generator, tokenizer, args, subtokenizer_id, logger)
        run_info = dict(task="translation", split=args.split, k_shots=args.k_shots,
                        step=step, subtokenizer_id=subtokenizer_id,
                        subtokenizer_id_out=args.subtokenizer_id_out,
                        target_lang=args.target_lang, reverse=args.reverse,
                        run_dir=args.run_dir, data_path=args.data_path, length=args.length)
        report_metrics(metrics, run_info, logger, args.output)


if __name__ == "__main__":
    main()
