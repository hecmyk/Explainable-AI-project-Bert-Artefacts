"""Shared helper utilities."""

import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 0) -> None:
    """Set deterministic seeds for Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_valid_token_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer) -> torch.Tensor:
    """
    Build a token mask that excludes padding and special tokens.

    Args:
        input_ids: (batch, seq_len)
        attention_mask: (batch, seq_len)
    """
    valid_mask = attention_mask.bool().clone()
    special_mask = torch.zeros_like(valid_mask)
    for token_id in tokenizer.all_special_ids:
        special_mask |= input_ids.eq(token_id)
    valid_mask &= ~special_mask
    return valid_mask


def to_jsonable(value: Any):
    """Convert tensors to JSON-serializable values."""
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    return value
