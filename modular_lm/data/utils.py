"""Utilities of the data pipeline: seed chains and file reading.

Seed chains — stateless, reproducible randomness.

A *chain* is a sequence of seeds where each is a fixed pseudo-random
permutation of the previous one (``next_seed``). A chain's starting point is
created deterministically from the training seed and the chain's identity (``create_seed``), so
every source of randomness is a pure function of (training seed, identity,
number of events) — nothing to replay, almost nothing to checkpoint.

A seed is consumed by seeding a throwaway ``random.Random(seed)`` for one
decision; continuity lives only in the chain.
"""
import io
import os

_MASK64 = (1 << 64) - 1


def next_seed(seed: int) -> int:
    """The next seed in a chain: a fixed pseudo-random permutation of the
    previous one (the splitmix64 mixer)."""
    x = seed
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def _stable_int64(part) -> int:
    """A process-stable 64-bit integer for a seed component (int or str) —
    never Python's built-in hash(), which is salted per process."""
    if isinstance(part, int):
        return part & _MASK64
    if isinstance(part, str):
        import hashlib

        return int.from_bytes(hashlib.blake2b(part.encode(), digest_size=8).digest(), "big")
    raise TypeError(f"cannot use {type(part)} as a seed component")


def create_seed(*components) -> int:
    """A chain's starting seed, computed deterministically from its identity —
    e.g. create_seed(training_seed, SUBTOKENIZER_SAMPLING, rank, lang)."""
    seed = 0
    for c in components:
        seed = next_seed(seed ^ _stable_int64(c))
    return seed


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------



def iter_file_lines(path: str):
    """Yield text lines from a file (.zst-compressed or plain)."""
    if path.endswith(".zst"):
        import zstandard as zstd

        with open(path, "rb") as f, zstd.ZstdDecompressor().stream_reader(f) as r:
            yield from io.TextIOWrapper(r, encoding="utf-8")
    else:
        with open(path, encoding="utf-8") as f:
            yield from f


def source_files(path: str) -> list[str]:
    """A source's file list: the file itself, or every regular file in a
    directory (sorted)."""
    if os.path.isdir(path):
        files = sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        )
        if not files:
            raise ValueError(f"no files in source directory {path}")
        return files
    return [path]
