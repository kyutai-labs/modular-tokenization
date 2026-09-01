"""Multiple-choice question evaluation (ARC, HellaSwag, MMLU, ...), scored
under a subtokenizer's vocabulary.

Two scoring modes:
  - cloze (default): each choice is scored as a continuation; reports the
    paper's three normalizations (unnormalized / length-normalized /
    prob-normalized cross-entropy);
  - letter: the model picks the answer letter (A/B/C/...) by next-token logits.

``data_path`` points at ONE task's own directory: ``train.jsonl`` +
``<split>.jsonl`` in the dataset's original format (for mmlu: the standard
``dev/ val/ test/`` CSV tree). ``task`` only selects the prompt format and the
file loader, never a path.

Run as ``python -m modular_lm.evaluation.mcq run_dir=... task=arc data_path=...
subtokenizer_id=en``.
"""

import csv
import json
import os
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice

import numpy as np
from pydantic import BaseModel, ConfigDict

from modular_lm.evaluation.utils import SequenceScorer, report_metrics
from modular_lm.training.checkpoint import load_model
from modular_lm.utils import Logger
from modular_tokenizers.config import as_str_list, parse_args_to_pydantic_model


# ---------------------------------------------------------------------------
# Dataset loaders — one per distribution format, yielding
# {question, choices, gold} (+ optional context)
# ---------------------------------------------------------------------------

def load_std(filename):
    with open(filename) as fin:
        for line in fin:
            data = json.loads(line)
            data["gold"] = data["answer"]
            yield data


def load_allenai(filename):
    with open(filename) as fin:
        for line in fin:
            data = json.loads(line)
            r = {"question": data["question"]["stem"], "choices": []}
            for i, c in enumerate(data["question"]["choices"]):
                r["choices"].append(c["text"])
                if c["label"] == data["answerKey"]:
                    r["gold"] = i
            if "gold" in r:
                yield r


def load_hellaswag(filename):
    with open(filename) as fin:
        for line in fin:
            data = json.loads(line)
            question = data["ctx"]
            if "activity_label" in data:
                question = data["activity_label"] + ": " + question
            yield {"question": question, "choices": data["endings"],
                   "gold": int(data["label"])}


def load_winogrande(filename):
    with open(filename) as fin:
        for line in fin:
            data = json.loads(line)
            yield {"question": data["end"], "choices": data["ctx"],
                   "gold": data["label"]}


def load_mmlu(path, split, topic):
    split = {"train": "dev", "valid": "val", "test": "test"}[split]
    filename = os.path.join(path, split, f"{topic}_{split}.csv")
    letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    with open(filename, newline="") as fin:
        for row in csv.reader(fin):
            yield {"question": row[0], "choices": row[1:5],
                   "gold": letter_to_idx[row[5]]}


# ---------------------------------------------------------------------------
# Task formats (the paper's prompts and default shot counts)
# ---------------------------------------------------------------------------

@dataclass
class Task:
    prefix: str
    prompt_format: str
    answer_format: str
    trainset: Iterable
    evalset: Iterable


QA_PREFIX = "The following is a list of elementary questions and answers."
QA_PROMPT = "Question: {q} Answer: {a}"
QA_ANSWER = "Answer: {a}"

# task -> (prefix, prompt_format, answer_format, loader, default k-shots)
TASK_FORMATS = {
    "arc": (QA_PREFIX, QA_PROMPT, QA_ANSWER, load_allenai, 5),
    "csqa": (QA_PREFIX, QA_PROMPT, QA_ANSWER, load_allenai, 5),
    "exams": (QA_PREFIX, QA_PROMPT, QA_ANSWER, load_allenai, 5),
    "belebele": (QA_PREFIX, QA_PROMPT, QA_ANSWER, load_allenai, 5),
    "obqa": (QA_PREFIX, "Question: {q} {a}", QA_ANSWER, load_allenai, 5),
    "siqa": (QA_PREFIX, QA_PROMPT, QA_ANSWER, load_hellaswag, 5),
    "piqa": ("The following is a list of elementary facts.", "{q} {a}",
             QA_ANSWER, load_hellaswag, 5),
    "global_piqa": ("The following is a list of elementary facts.",
                    "Goal: {q} Answer: {a}", QA_ANSWER, load_hellaswag, 0),
    "hellaswag": ("The following is a list of commonsense facts.", "{q} {a}",
                  QA_ANSWER, load_hellaswag, 0),
    "winogrande": ("The following is a list of elementary facts.", "{a}{q}",
                   "{a}", load_winogrande, 5),
    "sciq": ("", QA_PROMPT, QA_ANSWER, load_std, 5),
    "boolq": ("", QA_PROMPT, QA_ANSWER, load_std, 5),
    "mmlu": ("The following are multiple choice questions (with answers) about {topic}.",
             QA_PROMPT, QA_ANSWER, load_mmlu, 5),
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def eval_cloze_task(scorer, task, k, subtokenizer_id, max_examples=-1, logger=None):
    def format_example(example, answer):
        text = task.prompt_format.format(q=example["question"], a=answer)
        if "context" in example:
            text = example["context"] + "\n" + text
        return text

    shots = [format_example(e, e["choices"][e["gold"]]) for e in task.trainset]
    prompt = "\n".join([task.prefix] + shots[:k]) + "\n"

    metrics = defaultdict(float)
    evalset = islice(task.evalset, max_examples) if max_examples > 0 else task.evalset
    for example in evalset:
        sequences = [prompt + format_example(example, "").strip()]
        for choice in example["choices"]:
            sequences.append(prompt + format_example(example, choice).strip())
            sequences.append(task.answer_format.format(a=choice).strip())

        scores = scorer.scores(sequences, subtokenizer_id)
        prompt_score = scores[0]
        scores = scores[1:].reshape(-1, 2)   # (full sequence, answer alone)
        choice_chars = np.asarray([len(c) for c in example["choices"]])

        gold = example["gold"]
        metrics["unnormalized"] += scores[:, 0].argmin() == gold
        metrics["length_normalized"] += ((scores[:, 0] - prompt_score) / choice_chars).argmin() == gold
        metrics["prob_normalized"] += (scores[:, 0] - scores[:, 1]).argmin() == gold
        metrics["n_examples"] += 1

    for key in ("unnormalized", "length_normalized", "prob_normalized"):
        metrics[key] /= metrics["n_examples"]
    return metrics


def eval_letter_task(scorer, task, k, subtokenizer_id, max_examples=-1, logger=None):
    def format_example(e, with_gold=False):
        letters = list("ABCDEFGH")[: len(e["choices"])]
        choices = "\n".join(f"{x}. {text}" for text, x in zip(e["choices"], letters))
        text = f"{e['question']}\n{choices}\nAnswer:"
        if with_gold:
            text += " " + letters[e["gold"]]
        if "context" in e:
            text = e["context"] + "\n" + text
        return text

    shots = [format_example(e, with_gold=True) for e in task.trainset]
    prompt = "\n".join([task.prefix] + shots[:k]) + "\n"

    metrics = defaultdict(float)
    letters = list("ABCDEFGH")
    evalset = islice(task.evalset, max_examples) if max_examples > 0 else task.evalset
    for example in evalset:
        sequences = [prompt + "\n\n" + format_example(example)]
        logits = scorer.next_token_logits(sequences, letters, subtokenizer_id)
        metrics["acc"] += logits[0, : len(example["choices"])].argmax() == example["gold"]
        metrics["n_examples"] += 1

    metrics["acc"] /= metrics["n_examples"]
    return metrics


def eval_mmlu(scorer, data_path, split, k, subtokenizer_id, max_examples, cloze, logger):
    """Average the per-topic metrics over every topic present in ``dev/``."""
    prefix, prompt_format, answer_format, _, _ = TASK_FORMATS["mmlu"]
    topics = sorted(f[:-8] for f in os.listdir(os.path.join(data_path, "dev"))
                    if f.endswith("_dev.csv"))
    metrics = defaultdict(float)
    keys = ("unnormalized", "length_normalized", "prob_normalized") if cloze else ("acc",)
    for topic in topics:
        task = Task(
            prefix=prefix.format(topic=topic.replace("_", " ")),
            prompt_format=prompt_format,
            answer_format=answer_format,
            trainset=load_mmlu(data_path, "train", topic),
            evalset=load_mmlu(data_path, split, topic),
        )
        eval_task = eval_cloze_task if cloze else eval_letter_task
        m = eval_task(scorer, task, k, subtokenizer_id, max_examples)
        for key in keys:
            metrics[key] += m[key]
        metrics["n_topics"] += 1
        logger.log(f"{topic:<40}: " + " ".join(f"{m[key]:5.3f}" for key in keys))

    for key in keys:
        metrics[key] /= metrics["n_topics"]
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class McqArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_dir: str | None = None
    step: int = -1
    task: str | None = None              # one of TASK_FORMATS
    data_path: str | None = None         # this task's own directory
    split: str = "test"
    subtokenizer_ids: str | None = None  # comma-separated; None = full vocabulary
    k_shots: int | None = None           # None = the task's default
    max_examples: int = -1
    cloze: bool = True
    flash_attention: bool = False
    output: str | None = None            # append metrics as JSON lines


def main():
    args = parse_args_to_pydantic_model(McqArgs)
    assert args.run_dir and args.task and args.data_path, \
        "run_dir, task and data_path are required"
    assert args.task in TASK_FORMATS, f"unknown task: {args.task} (choose from {sorted(TASK_FORMATS)})"
    logger = Logger()
    logger.log(json.dumps(args.model_dump()))

    model, params, tokenizer, step = load_model(
        args.run_dir, args.step, flash_attention=args.flash_attention
    )
    scorer = SequenceScorer(model, params, tokenizer)

    prefix, prompt_format, answer_format, loader, default_k = TASK_FORMATS[args.task]
    k = args.k_shots if args.k_shots is not None else default_k
    subtokenizer_ids = as_str_list(args.subtokenizer_ids) if args.subtokenizer_ids else [None]

    for subtokenizer_id in subtokenizer_ids:
        if args.task == "mmlu":
            metrics = eval_mmlu(scorer, args.data_path, args.split, k,
                                subtokenizer_id, args.max_examples, args.cloze, logger)
        else:
            task = Task(
                prefix=prefix, prompt_format=prompt_format, answer_format=answer_format,
                trainset=loader(os.path.join(args.data_path, "train.jsonl")),
                evalset=loader(os.path.join(args.data_path, f"{args.split}.jsonl")),
            )
            eval_task = eval_cloze_task if args.cloze else eval_letter_task
            metrics = eval_task(scorer, task, k, subtokenizer_id, args.max_examples)

        run_info = dict(task=args.task, split=args.split, k_shots=k, step=step,
                        subtokenizer_id=subtokenizer_id, run_dir=args.run_dir,
                        data_path=args.data_path, cloze=args.cloze)
        report_metrics(dict(metrics), run_info, logger, args.output)


if __name__ == "__main__":
    main()
