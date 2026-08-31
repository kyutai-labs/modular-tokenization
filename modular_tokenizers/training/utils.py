"""Helpers shared by the BPE and Unigram training pipelines.

Parts adapted from tokenizer-extension (Purason et al.,
https://github.com/taidopurason/tokenizer-extension, Apache-2.0 — see the
Licenses section of the README).
"""
import json
import os

from tokenizers import Tokenizer as HFTokenizer

SPECIAL_TOKENS = ["<s>", "</s>", "<pad>", "<unk>"]

# SentencePiece-style byte-fallback pieces (<0x00> ... <0xFF>).
BYTE_PIECES = [f"<0x{i:02X}>" for i in range(256)]


def get_hf_json(tokenizer) -> dict:
    """Full tokenizer.json config of an HF tokenizer (raw or
    ``PreTrainedTokenizerFast`` wrapper) as a dict."""
    raw = getattr(tokenizer, "_tokenizer", tokenizer)
    return json.loads(raw.to_str())


def hf_from_json(cfg: dict) -> HFTokenizer:
    return HFTokenizer.from_str(json.dumps(cfg))


def load_data(path, maxload=-1):
    """Read a text corpus: one document per line; blank lines glue the next
    line onto the previous document (paragraph continuation)."""
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                texts.append(line)
            else:
                if texts:
                    texts[-1] = texts[-1] + line
            if len(texts) == maxload:
                break
    return texts


# ---------------------------------------------------------------------------
# Generic corpus reading — no assumption about how the user organizes data.
# A "source" is any path the user provides: a plain text file, a .zst file, or
# a directory (all regular files inside are read; .zst files are decompressed).
# ---------------------------------------------------------------------------

def iter_lines(source: str):
    """Yield raw text lines from one source path."""
    import shutil
    import subprocess

    if os.path.isdir(source):
        files = sorted(
            os.path.join(source, f)
            for f in os.listdir(source)
            if os.path.isfile(os.path.join(source, f))
        )
        if not files:
            raise ValueError(f"No files found in directory {source}")
    else:
        files = [source]

    zst_files = [f for f in files if f.endswith(".zst")]
    plain_files = [f for f in files if not f.endswith(".zst")]

    for f in plain_files:
        with open(f, "r", encoding="utf-8") as fh:
            yield from fh

    if zst_files:
        zstdcat = shutil.which("zstdcat") or "zstdcat"
        p = subprocess.Popen([zstdcat] + zst_files, stdout=subprocess.PIPE)
        for line in p.stdout:
            yield line.decode("utf-8")


def iter_docs(source: str, text_field: str | None = None, max_docs: int = -1):
    """Yield documents from one source: one document per non-empty line.
    ``text_field`` extracts that field from JSONL lines; otherwise lines are
    used raw."""
    n = 0
    for line in iter_lines(source):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        yield json.loads(line)[text_field] if text_field else line
        n += 1
        if n == max_docs:
            return


def interleave_docs(
    sources: list[str],
    weights: list[float] | None = None,
    text_field: str | None = None,
    seed: int = 1234,
):
    """Interleave documents from several sources, sampling each next document's
    source with the given weights (uniform if None). Exhausted sources drop out
    and the remaining weights are renormalized."""
    import random

    iters = {i: iter_docs(s, text_field) for i, s in enumerate(sources)}
    w = {i: (weights[i] if weights else 1.0) for i in iters}
    rng = random.Random(seed)

    while iters:
        keys = list(iters.keys())
        idx = rng.choices(keys, weights=[w[k] for k in keys])[0]
        try:
            yield next(iters[idx])
        except StopIteration:
            del iters[idx]


def read_json(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        return json.load(file)


def write_json(data, file_path, indent=4):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=indent)
