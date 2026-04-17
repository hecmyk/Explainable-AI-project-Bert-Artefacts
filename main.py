"""Paper-faithful reproduction pipeline for BERT-base (Sections 2 and 4.1)."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.clipping import clip_vector_lookup
from src.extract_embeddings import (
    build_token_metadata,
    iter_hidden_states,
    load_sentences_from_file,
    sample_sentence_indices,
    select_sentences,
)
from src.load_model import load_model
from src.outlier_analysis import (
    dominant_dimension_and_percentage,
    finalize_average_vectors,
    initialize_layer_statistics,
    select_outlier_dimensions,
    update_layer_statistics,
)
from src.similarity import (
    adjusted_self_similarity,
    compute_anisotropy_per_layer,
    compute_self_similarity_per_layer,
    sample_anisotropy_pairs,
    sample_self_similarity_contexts,
    select_self_similarity_words,
)
from src.utils import build_valid_token_mask, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic sampling.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for model forward passes.")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum number of tokens per sentence.")
    parser.add_argument("--figure1-sentences", type=int, default=10000, help="Number of SST-2 sentences for Section 2.")
    parser.add_argument(
        "--anisotropy-sentences",
        type=int,
        default=2000,
        help="Number of sampled sentences for anisotropy protocol (Section 4.1).",
    )
    parser.add_argument(
        "--anisotropy-pairs",
        type=int,
        default=1000,
        help="Number of sentence/token pairs for anisotropy (Section 4.1).",
    )
    parser.add_argument("--selfsim-words", type=int, default=1000, help="Number of words for self-similarity.")
    parser.add_argument(
        "--selfsim-min-sentences",
        type=int,
        default=10,
        help="Minimum sentence frequency for eligible self-similarity words.",
    )
    parser.add_argument(
        "--selfsim-contexts-per-word",
        type=int,
        default=10,
        help="Number of contexts sampled per self-similarity word.",
    )
    parser.add_argument(
        "--clip-top-k",
        type=int,
        default=1,
        help="Top-k outlier dimensions to clip per layer.",
    )
    parser.add_argument(
        "--clip-source",
        choices=["argmin", "argmax"],
        default="argmin",
        help="Histogram source used to select clipping dimensions.",
    )
    parser.add_argument(
        "--include-special-tokens",
        action="store_true",
        help="Allow [CLS]/[SEP] in Figure 6 random token/context selection.",
    )
    return parser.parse_args()


def build_required_positions(
    anisotropy_pairs: Sequence[Tuple[int, int, int, int]],
    word_contexts: Dict[str, List[Tuple[int, int]]],
) -> Dict[int, List[int]]:
    """Collect all (sentence, token_position) pairs needed for Figure 6 metrics."""
    required: Dict[int, set] = defaultdict(set)

    for sent_a, pos_a, sent_b, pos_b in anisotropy_pairs:
        required[sent_a].add(pos_a)
        required[sent_b].add(pos_b)

    for contexts in word_contexts.values():
        for sent_idx, token_pos in contexts:
            required[sent_idx].add(token_pos)

    return {sent_idx: sorted(list(pos_set)) for sent_idx, pos_set in required.items()}


def collect_statistics_and_vectors(
    sentences: Sequence[str],
    tokenizer,
    model,
    batch_size: int,
    max_length: int,
    required_positions: Dict[int, List[int]],
):
    """Run BERT once and collect stats plus vectors ."""
    num_layers = model.config.num_hidden_layers + 1
    hidden_dim = model.config.hidden_size

    layer_sums, token_counts, argmin_counts, argmax_counts = initialize_layer_statistics(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
    )
    vector_lookup: Dict[Tuple[int, int], torch.Tensor] = {}
    missing_positions: List[Tuple[int, int]] = []

    for hidden_states, input_ids, attention_mask, start_idx in iter_hidden_states(
        sentences=sentences,
        tokenizer=tokenizer,
        model=model,
        batch_size=batch_size,
        max_length=max_length,
    ):
        valid_mask = build_valid_token_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
        )

        update_layer_statistics(
            hidden_states=hidden_states,
            valid_mask=valid_mask,
            layer_sums=layer_sums,
            token_counts=token_counts,
            argmin_counts=argmin_counts,
            argmax_counts=argmax_counts,
        )

        batch_size_now = input_ids.size(0)
        seq_len = input_ids.size(1)

        for local_idx in range(batch_size_now):
            sentence_idx = start_idx + local_idx
            sentence_positions = required_positions.get(sentence_idx)
            if not sentence_positions:
                continue

            for token_pos in sentence_positions:
                if token_pos >= seq_len or attention_mask[local_idx, token_pos].item() == 0:
                    missing_positions.append((sentence_idx, token_pos))
                    continue
                vector_lookup[(sentence_idx, token_pos)] = hidden_states[:, local_idx, token_pos, :].clone()

    if missing_positions:
        raise ValueError(
            f"Could not retrieve {len(missing_positions)} sampled positions after tokenization/truncation. "
            "Increase --max-length or regenerate samples."
        )

    return layer_sums, token_counts, argmin_counts, argmax_counts, vector_lookup


def plot_figure1_average_vectors(average_vectors: torch.Tensor, output_path: Path) -> None:
    """Plot average contextualized vector per layer."""
    num_layers = average_vectors.size(0)

    fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
    flat_axes = axes.flatten()

    for layer_idx in range(num_layers):
        ax = flat_axes[layer_idx]
        ax.plot(average_vectors[layer_idx].numpy(), linewidth=1.0, color="#2E6F95")
        ax.set_title(f"Layer {layer_idx}")
        ax.set_ylabel("Mean value")
        ax.grid(alpha=0.2)

    for ax in flat_axes[num_layers:]:
        ax.axis("off")

    for ax in flat_axes[-4:]:
        ax.set_xlabel("Embedding dimension")

    fig.suptitle("Average contextualized vector per layer (BERT-base)", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_outlier_heatmap(counts: torch.Tensor, title: str, output_path: Path) -> None:
    """Plot layer-by-dimension outlier histogram heatmap."""
    plt.figure(figsize=(14, 5))
    plt.imshow(counts.numpy(), aspect="auto", cmap="viridis")
    plt.colorbar(label="Count")
    plt.xlabel("Embedding dimension")
    plt.ylabel("Layer")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_figure6_curves(
    before_values: Sequence[float],
    after_values: Sequence[float],
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """layer curves before/after clipping."""
    layers = list(range(len(before_values)))
    plt.figure(figsize=(10, 5))
    plt.plot(layers, before_values, marker="o", label="Before clipping", color="#C44536")
    plt.plot(layers, after_values, marker="o", label="After clipping", color="#3A7D44")
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    project_root = Path(__file__).resolve().parent
    results_dir = project_root / "results"
    plots_dir = results_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Loading BERT-base...")
    tokenizer, model = load_model("bert-base-uncased")

    sentences_path = project_root / "data" / "sentences.txt"
    print(f"Loading local sentences from {sentences_path} ...")
    all_sentences = load_sentences_from_file(str(sentences_path))

    print(f"Sampling {args.figure1_sentences} sentences for Section 2 / Figure 1...")
    figure_source_indices = sample_sentence_indices(
        total_size=len(all_sentences),
        sample_size=args.figure1_sentences,
        seed=args.seed,
    )
    figure_sentences = select_sentences(all_sentences, figure_source_indices)

    print("Building token metadata for deterministic sampling...")
    (
        _input_ids_per_sentence,
        valid_positions_per_sentence,
        word_occurrences,
        word_sentence_sets,
    ) = build_token_metadata(
        sentences=figure_sentences,
        tokenizer=tokenizer,
        max_length=args.max_length,
        include_special_tokens=args.include_special_tokens,
    )

    print(f"Sampling Section 4.1 anisotropy protocol ({args.anisotropy_sentences} sentences, {args.anisotropy_pairs} pairs)...")
    anisotropy_sentence_indices = sample_sentence_indices(
        total_size=len(figure_sentences),
        sample_size=args.anisotropy_sentences,
        seed=args.seed + 1,
    )
    anisotropy_pairs = sample_anisotropy_pairs(
        sentence_indices=anisotropy_sentence_indices,
        valid_positions_per_sentence=valid_positions_per_sentence,
        num_pairs=args.anisotropy_pairs,
        seed=args.seed + 2,
    )

    print(f"Sampling self-similarity protocol ({args.selfsim_words} words, >= {args.selfsim_min_sentences} sentences)...")
    selected_words = select_self_similarity_words(
        word_sentence_sets=word_sentence_sets,
        num_words=args.selfsim_words,
        min_sentences=args.selfsim_min_sentences,
        seed=args.seed + 3,
    )
    word_contexts = sample_self_similarity_contexts(
        word_occurrences=word_occurrences,
        selected_words=selected_words,
        contexts_per_word=args.selfsim_contexts_per_word,
        seed=args.seed + 4,
    )

    required_positions = build_required_positions(anisotropy_pairs, word_contexts)

    print("Running BERT and collecting per-layer statistics/vectors...")
    layer_sums, token_counts, argmin_counts, argmax_counts, vector_lookup = collect_statistics_and_vectors(
        sentences=figure_sentences,
        tokenizer=tokenizer,
        model=model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        required_positions=required_positions,
    )

    average_vectors = finalize_average_vectors(layer_sums, token_counts)
    num_layers = average_vectors.size(0)

    dominant_argmin_dims, dominant_argmin_pct = dominant_dimension_and_percentage(argmin_counts)
    dominant_argmax_dims, dominant_argmax_pct = dominant_dimension_and_percentage(argmax_counts)

    selected_dims_per_layer = select_outlier_dimensions(
        argmin_counts=argmin_counts,
        argmax_counts=argmax_counts,
        top_k=args.clip_top_k,
        source=args.clip_source,
    )

    clipped_vector_lookup = clip_vector_lookup(vector_lookup, selected_dims_per_layer)

    anisotropy_before = compute_anisotropy_per_layer(
        vector_lookup=vector_lookup,
        pairs=anisotropy_pairs,
        num_layers=num_layers,
    )
    anisotropy_after = compute_anisotropy_per_layer(
        vector_lookup=clipped_vector_lookup,
        pairs=anisotropy_pairs,
        num_layers=num_layers,
    )

    self_similarity_before = compute_self_similarity_per_layer(
        vector_lookup=vector_lookup,
        word_contexts=word_contexts,
        num_layers=num_layers,
    )
    self_similarity_after = compute_self_similarity_per_layer(
        vector_lookup=clipped_vector_lookup,
        word_contexts=word_contexts,
        num_layers=num_layers,
    )

    adjusted_self_before = adjusted_self_similarity(self_similarity_before, anisotropy_before)
    adjusted_self_after = adjusted_self_similarity(self_similarity_after, anisotropy_after)

    special_tag = "withspecial" if args.include_special_tokens else "nospecial"
    run_tag = f"repro_seed{args.seed}_topk{args.clip_top_k}_{args.clip_source}_{special_tag}"

    figure1_plot = plots_dir / f"average_vectors_{run_tag}.png"
    argmin_plot = plots_dir / f"argmin_histogram_{run_tag}.png"
    argmax_plot = plots_dir / f"argmax_histogram_{run_tag}.png"
    anisotropy_plot = plots_dir / f"anisotropy_{run_tag}.png"
    adjusted_self_plot = plots_dir / f"adjusted_self_similarity_{run_tag}.png"

    plot_figure1_average_vectors(average_vectors, figure1_plot)
    plot_outlier_heatmap(argmin_counts, "argmin frequency per layer", argmin_plot)
    plot_outlier_heatmap(argmax_counts, "argmax frequency per layer", argmax_plot)
    plot_figure6_curves(
        before_values=anisotropy_before,
        after_values=anisotropy_after,
        ylabel="Average cosine similarity",
        title="Anisotropy per layer",
        output_path=anisotropy_plot,
    )
    plot_figure6_curves(
        before_values=adjusted_self_before,
        after_values=adjusted_self_after,
        ylabel="Adjusted self-similarity",
        title="Adjusted self-similarity per layer",
        output_path=adjusted_self_plot,
    )

    metrics = {
        "model_name": "bert-base-uncased",
        "metadata": {
            "seed": args.seed,
            "clip_top_k": args.clip_top_k,
            "clip_source": args.clip_source,
            "include_special_tokens": bool(args.include_special_tokens),
        },
        "outputs": {
            "dominant_argmin_dimension_per_layer": dominant_argmin_dims,
            "dominant_argmax_dimension_per_layer": dominant_argmax_dims,
            "anisotropy_before_per_layer": anisotropy_before,
            "anisotropy_after_per_layer": anisotropy_after,
            "adjusted_self_similarity_before_per_layer": adjusted_self_before,
            "adjusted_self_similarity_after_per_layer": adjusted_self_after,
        },
    }

    metrics_path = results_dir / f"metrics_{run_tag}.json"
    latest_metrics_path = results_dir / "metrics.json"
    metrics_json = json.dumps(metrics, indent=2)
    metrics_path.write_text(metrics_json, encoding="utf-8")
    latest_metrics_path.write_text(metrics_json, encoding="utf-8")

    print("\nSection 2 dominant argmin dimensions (layer -> dim, %):")
    for layer_idx in range(num_layers):
        print(f"  L{layer_idx:02d}: dim {dominant_argmin_dims[layer_idx]} ({dominant_argmin_pct[layer_idx]:.2f}%)")

    print("\nFigure 6 metrics (layer 0 and last layer):")
    last = num_layers - 1
    print(
        f"  Anisotropy L00/L{last}: "
        f"{anisotropy_before[0]:.6f}->{anisotropy_after[0]:.6f} / "
        f"{anisotropy_before[last]:.6f}->{anisotropy_after[last]:.6f}"
    )
    print(
        f"  Adjusted self-sim L00/L{last}: "
        f"{adjusted_self_before[0]:.6f}->{adjusted_self_after[0]:.6f} / "
        f"{adjusted_self_before[last]:.6f}->{adjusted_self_after[last]:.6f}"
    )

    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved average vectors plot: {figure1_plot}")
    print(f"Saved outlier histograms: {argmin_plot}, {argmax_plot}")
    print(f"Saved anisotropy/self-similarity plots: {anisotropy_plot}, {adjusted_self_plot}")


if __name__ == "__main__":
    main()
