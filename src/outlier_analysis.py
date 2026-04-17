"""Layer-wise outlier statistics for contextualized token vectors."""

from typing import List, Tuple

import torch


def initialize_layer_statistics(num_layers: int, hidden_dim: int):
    """Initialize accumulators for average vectors and outlier histograms."""
    layer_sums = torch.zeros(num_layers, hidden_dim, dtype=torch.float32)
    token_counts = torch.zeros(num_layers, dtype=torch.long)
    argmin_counts = torch.zeros(num_layers, hidden_dim, dtype=torch.long)
    argmax_counts = torch.zeros(num_layers, hidden_dim, dtype=torch.long)
    return layer_sums, token_counts, argmin_counts, argmax_counts


def update_layer_statistics(
    hidden_states: torch.Tensor,
    valid_mask: torch.Tensor,
    layer_sums: torch.Tensor,
    token_counts: torch.Tensor,
    argmin_counts: torch.Tensor,
    argmax_counts: torch.Tensor,
) -> None:
    """
    Update per-layer stats from one batch.

    Args:
        hidden_states: (num_layers, batch, seq_len, hidden_dim)
        valid_mask: (batch, seq_len) boolean mask for valid tokens
    """
    num_layers = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    for layer_idx in range(num_layers):
        layer_vectors = hidden_states[layer_idx][valid_mask]
        if layer_vectors.numel() == 0:
            continue

        layer_sums[layer_idx] += layer_vectors.sum(dim=0)
        token_counts[layer_idx] += layer_vectors.size(0)

        layer_argmin = torch.argmin(layer_vectors, dim=1)
        layer_argmax = torch.argmax(layer_vectors, dim=1)

        argmin_counts[layer_idx] += torch.bincount(layer_argmin, minlength=hidden_dim)
        argmax_counts[layer_idx] += torch.bincount(layer_argmax, minlength=hidden_dim)


def finalize_average_vectors(layer_sums: torch.Tensor, token_counts: torch.Tensor) -> torch.Tensor:
    """Compute average vector for each layer."""
    averages = layer_sums.clone()
    for layer_idx in range(layer_sums.size(0)):
        if token_counts[layer_idx] > 0:
            averages[layer_idx] /= token_counts[layer_idx].float()
    return averages


def dominant_dimension_and_percentage(counts: torch.Tensor) -> Tuple[List[int], List[float]]:
    """
    For each layer, return dominant dimension and percentage share.

    Args:
        counts: (num_layers, hidden_dim)
    """
    dominant_dims = torch.argmax(counts, dim=1)
    totals = counts.sum(dim=1).clamp(min=1)
    percentages = 100.0 * counts[torch.arange(counts.size(0)), dominant_dims].float() / totals.float()
    return dominant_dims.tolist(), percentages.tolist()


def select_outlier_dimensions(
    argmin_counts: torch.Tensor,
    argmax_counts: torch.Tensor,
    top_k: int = 1,
    source: str = "argmin",
) -> List[List[int]]:
    """
    Select outlier dimensions per layer from argmin or argmax histograms.
    """
    if source not in {"argmin", "argmax"}:
        raise ValueError("source must be 'argmin' or 'argmax'.")

    counts = argmin_counts if source == "argmin" else argmax_counts
    top_k = max(1, min(top_k, counts.size(1)))

    selected = []
    for layer_idx in range(counts.size(0)):
        dims = torch.topk(counts[layer_idx], k=top_k).indices.tolist()
        selected.append(dims)
    return selected
