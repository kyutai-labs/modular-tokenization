# Modular Tokenization

Code for the paper *To Each Language Its Tokenizer: Modular Tokenizers for
Efficient Multilingual LLMs* (EMNLP 2026): learn large modular BPE and
Unigram tokenizers whose per-language subtokenizers can be extracted and
composed for any language subset, and train multilingual LLMs whose
predictions are restricted to the sampled sub-vocabulary.

> README under construction — installation, reproduction commands and the
> folder structure will be completed before release.

## Installation

```bash
git clone <repo-url>
cd modular-tokenization
uv sync
```

## Folder Structure

```
modular_tokenizers/        # building & using modular tokenizers (no GPU deps)
├── training/
│   ├── bpe/               # sequential BPE (paper §3.2) + serialization
│   └── unigram/           # merged Unigram: train, merge, EM re-estimation (paper §3.1)
├── subtokenizers/         # loading, per-language masks, extraction, composition sampling
└── evaluation/            # NSL (paper Eq. 1)
```

## Licenses

The present code is provided under the MIT license.

Portions of `modular_tokenizers/training/bpe/continued_training.py` and
`modular_tokenizers/training/utils.py` are adapted from
[tokenizer-extension](https://github.com/taidopurason/tokenizer-extension)
(Taido Purason et al., *Teaching Old Tokenizers New Words*), released under
the Apache License 2.0; the corresponding attribution is kept in those files'
headers.

## Citation

```bibtex
(to be added upon publication)
```
