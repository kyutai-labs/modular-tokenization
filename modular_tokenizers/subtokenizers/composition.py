"""Composition sampling: which subtokenizer tokenizes each training batch.

A *composition* is a subset of languages identified by a sorted comma-joined
id (e.g. ``"en,fr"``; ``"all"`` is the full tokenizer). During pretraining,
each batch in language L is tokenized either with L's own subtokenizer or
with a sampled composition containing L (paper: the Full and Data(n)
strategies).
"""
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, field_serializer, model_validator


class SamplingMethod(Enum):
    FULL = "full"
    DATA_WEIGHTED = "data_weighted"
    UNIFORM = "uniform"


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: SamplingMethod | None = None
    p_lang: float | None = None
    n_sampled_subtokenizers: int | None = None
    n_extra_langs: int | None = None
    main_lang: str | None = None
    p_main_lang: float | None = None

    @model_validator(mode="after")
    def validate_strategy_args(self):
        """
        Ensures p_lang is present if any strategy is selected,
        and validates specific requirements for DATA_WEIGHTED.
        """

        if (self.strategy is not None) != (self.p_lang is not None):
            raise ValueError(
                "Both 'strategy' and 'p_lang' must be provided together. "
                f"Current state: strategy={self.strategy}, p_lang={self.p_lang}"
            )

        if self.strategy == SamplingMethod.DATA_WEIGHTED:
            required = ["p_lang", "main_lang", "p_main_lang", "n_sampled_subtokenizers", "n_extra_langs"]
            missing = [f for f in required if getattr(self, f) is None]

            if missing:
                raise ValueError(f"DATA_WEIGHTED strategy requires: {', '.join(missing)}")

        if self.strategy == SamplingMethod.UNIFORM:
            # Same as DATA_WEIGHTED but extra languages are drawn uniformly, so no
            # main-language bias: main_lang / p_main_lang must be omitted.
            required = ["p_lang", "n_sampled_subtokenizers", "n_extra_langs"]
            missing = [f for f in required if getattr(self, f) is None]
            if missing:
                raise ValueError(f"UNIFORM strategy requires: {', '.join(missing)}")
            assert self.main_lang is None and self.p_main_lang is None, (
                "UNIFORM strategy draws extra languages uniformly; "
                "main_lang and p_main_lang must be None."
            )

        return self

    @field_serializer('strategy')
    def serialize_strategy(self, strategy: SamplingMethod | None):
        return strategy.value if strategy else None

    def _get_weighted_samples(self, current_lang: str, all_langs: List[str]) -> List[str]:
        """Internal helper to handle the NumPy sampling logic."""
        candidates = [l for l in all_langs if l != current_lang]
        if not candidates:
            return []

        n_to_sample = min(self.n_extra_langs, len(candidates))

        if self.strategy == SamplingMethod.UNIFORM:
            # Every candidate language equiprobable (no main-language bias).
            probs = np.ones(len(candidates), dtype=float)
        else:
            other_candidates = [l for l in candidates if l != self.main_lang]
            num_others = len(other_candidates)
            probs = []
            for l in candidates:
                if l == self.main_lang:
                    probs.append(self.p_main_lang)
                else:
                    p_other = (1.0 - self.p_main_lang) / num_others if num_others > 0 else 0
                    probs.append(p_other)
            probs = np.array(probs, dtype=float)

        probs /= probs.sum()

        # Draw up to n_sampled_subtokenizers compositions and keep the UNIQUE ones
        # (order-preserving). Sampling is with replacement, so without dedup the
        # repeats inflate n_extra in build_compositions_by_lang while probs (a
        # dict) collapses them -> mixed-composition weights would not sum to
        # (1 - p_lang). Dedup keeps compositions / n_extra / probs consistent.
        seen, groups = set(), []
        for _ in range(self.n_sampled_subtokenizers):
            sampled = np.random.choice(candidates, size=n_to_sample, replace=False, p=probs)
            group_name = ",".join(sorted([current_lang] + list(sampled)))
            if group_name not in seen:
                seen.add(group_name)
                groups.append(group_name)
        return groups

    def build_compositions_by_lang(self, langs: List[str]) -> Tuple[Dict, Dict | None]:
        """For each language: the composition ids its batches may use, and their
        sampling probabilities -> ({lang: [composition_id]}, {lang: {composition_id: p}})."""
        compositions = {}
        probs = {}

        for src in langs:
            if self.strategy is None:
                # No sampling: every language uses the full (merged) vocab.
                comps = ['all']
            elif self.strategy == SamplingMethod.FULL:
                comps = [src, 'all']
            else:
                comps = [src] + self._get_weighted_samples(src, langs)

            compositions[src] = comps

            if self.p_lang is not None:
                n_extra = len(comps) - 1
                p_extra = (1.0 - self.p_lang) / n_extra if n_extra > 0 else 0.0
                probs[src] = {c: (self.p_lang if c == src else p_extra) for c in comps}

        return compositions, probs
