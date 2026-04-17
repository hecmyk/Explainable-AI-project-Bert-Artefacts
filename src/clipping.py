"""Layer-wise clipping helpers."""

from typing import Dict, List, Tuple

import torch


def clip_layerwise_tensor(tensor: torch.Tensor, selected_dims_per_layer: List[List[int]]) -> torch.Tensor:
    """
    Clip selected dimensions for each layer.

    Args:
        tensor: (..., hidden_dim) with first dimension = layer index
                expected shape here: (num_layers, hidden_dim)
        selected_dims_per_layer: list of dimensions to clip for each layer
    """
    clipped = tensor.clone()
    for layer_idx, dims in enumerate(selected_dims_per_layer):
        if dims:
            clipped[layer_idx, dims] = 0.0
    return clipped


def clip_vector_lookup(
    vector_lookup: Dict[Tuple[int, int], torch.Tensor],
    selected_dims_per_layer: List[List[int]],
) -> Dict[Tuple[int, int], torch.Tensor]:
    """
    Apply layer-wise clipping to all sampled vectors.

    vector_lookup maps (sentence_idx, token_pos) -> (num_layers, hidden_dim)
    """
    clipped_lookup: Dict[Tuple[int, int], torch.Tensor] = {}
    for key, vectors in vector_lookup.items():
        clipped_lookup[key] = clip_layerwise_tensor(vectors, selected_dims_per_layer)
    return clipped_lookup
