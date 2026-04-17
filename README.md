# BERT-base Positional Artifact Reproduction

This project reproduces the BERT-base parts of:

`Positional Artefacts Propagate Through Masked Language Model Embeddings`

Targeted outputs:

1. average vectors per layer
2. outlier statistics (argmin/argmax histograms per layer)
3. anisotropy before/after clipping (per layer)
4. adjusted self-similarity before/after clipping (per layer)

## Protocol implemented

- Model: `bert-base-uncased` with `output_hidden_states=True`
- Sentence source: local file `data/sentences.txt` (SST-2 train sentences)
- Deterministic sampling with fixed seed

For average vectors and outlier histograms:
- deterministic sample of `10,000` sentences
- token-level contextualized vectors (special tokens excluded)

For anisotropy and adjusted self-similarity:
- deterministic sample of `2,000` sentences
- exactly `1,000` sentence pairs
- one random token per sentence in each pair
- same sampled pairs/contexts before and after clipping

## Install

```bash
pip install torch transformers matplotlib
```

## Run

```bash
python main.py
```

Useful options:

```bash
python main.py --clip-top-k 1 --clip-source argmin
python main.py --clip-top-k 5 --clip-source argmin
python main.py --batch-size 16 --max-length 128
python main.py --include-special-tokens
```

## Outputs

- Metrics JSON:
  - `results/metrics.json` (latest run)
  - `results/metrics_repro_seed<seed>_topk<k>_<source>_<nospecial|withspecial>.json`
- Plots:
  - `results/plots/average_vectors_...png`
  - `results/plots/argmin_histogram_...png`
  - `results/plots/argmax_histogram_...png`
  - `results/plots/anisotropy_...png`
  - `results/plots/adjusted_self_similarity_...png`
