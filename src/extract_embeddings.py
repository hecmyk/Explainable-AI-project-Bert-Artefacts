"""Data loading and hidden-state extraction helpers."""

from collections import defaultdict
from pathlib import Path
import random
from typing import Dict, Iterator, List, Sequence, Tuple

import torch


def load_sentences_from_file(sentences_path: str) -> List[str]:
    """Load one sentence per line from a local text file."""
    path = Path(sentences_path)
    if not path.exists():
        raise FileNotFoundError(f"Sentences file not found: {sentences_path}")

    sentences = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not sentences:
        raise ValueError(f"No sentences found in {sentences_path}")
    return sentences


def sample_sentence_indices(total_size: int, sample_size: int, seed: int) -> List[int]:
    """Sample unique sentence indices deterministically."""
    if sample_size > total_size:
        raise ValueError(f"Cannot sample {sample_size} items from {total_size}.")
    rng = random.Random(seed)
    return rng.sample(list(range(total_size)), sample_size)


def select_sentences(sentences: Sequence[str], indices: Sequence[int]) -> List[str]:
    """Select sentences by index."""
    return [sentences[i] for i in indices]


def build_token_metadata(
    sentences: Sequence[str],
    tokenizer,
    max_length: int,
    include_special_tokens: bool = False,
) -> Tuple[
    List[List[int]],
    List[List[int]],
    Dict[str, List[Tuple[int, int]]],
    Dict[str, set],
]:
    """
    Tokenize sentences (without padding) and build token occurrence metadata.

    Returns:
        input_ids_per_sentence: token ids per sentence
        valid_positions_per_sentence: non-special token positions per sentence
        word_occurrences: token -> list of (sentence_idx, token_pos), excluding special tokens and ## continuations
        word_sentence_sets: token -> set(sentence_idx)
    """
    special_ids = set(tokenizer.all_special_ids)
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    input_ids_per_sentence: List[List[int]] = []
    valid_positions_per_sentence: List[List[int]] = []
    word_occurrences: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    word_sentence_sets: Dict[str, set] = defaultdict(set)

    for sentence_idx, sentence in enumerate(sentences):
        encoded = tokenizer(
            sentence,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded["input_ids"]
        input_ids_per_sentence.append(input_ids)

        valid_positions = []
        seen_words_in_sentence = set()

        for token_pos, token_id in enumerate(input_ids):
            if token_id in special_ids:
                if include_special_tokens and token_id in {cls_id, sep_id}:
                    valid_positions.append(token_pos)
                    token = tokenizer.convert_ids_to_tokens(token_id)
                    word_occurrences[token].append((sentence_idx, token_pos))
                    seen_words_in_sentence.add(token)
                continue

            valid_positions.append(token_pos)

            token = tokenizer.convert_ids_to_tokens(token_id)
            if token.startswith("##"):
                continue
            if not token.isalpha():
                continue

            word_occurrences[token].append((sentence_idx, token_pos))
            seen_words_in_sentence.add(token)

        valid_positions_per_sentence.append(valid_positions)
        for token in seen_words_in_sentence:
            word_sentence_sets[token].add(sentence_idx)

    return (
        input_ids_per_sentence,
        valid_positions_per_sentence,
        word_occurrences,
        word_sentence_sets,
    )


def iter_hidden_states(
    sentences: Sequence[str],
    tokenizer,
    model,
    batch_size: int = 32,
    max_length: int = 128,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    """
    Yield BERT hidden states batch-by-batch.

    Yields:
        hidden_states: (num_layers, batch, seq_len, hidden_dim)
        input_ids: (batch, seq_len)
        attention_mask: (batch, seq_len)
        start_idx: first global sentence index in this batch
    """
    device = next(model.parameters()).device

    with torch.no_grad():
        for start_idx in range(0, len(sentences), batch_size):
            batch_sentences = sentences[start_idx : start_idx + batch_size]
            encoded = tokenizer(
                batch_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = torch.stack(outputs.hidden_states, dim=0).cpu()

            yield hidden_states, input_ids.cpu(), attention_mask.cpu(), start_idx
