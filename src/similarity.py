"""Section 4.1 style anisotropy and self-similarity metrics."""

from collections import defaultdict
import random
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


AnisotropyPair = Tuple[int, int, int, int]  # (sent_a, pos_a, sent_b, pos_b)


def sample_anisotropy_pairs(
    sentence_indices: Sequence[int],
    valid_positions_per_sentence: Sequence[Sequence[int]],
    num_pairs: int,
    seed: int,
) -> List[AnisotropyPair]:
    """
    Sample sentence pairs and one token per sentence (deterministically).

    Protocol:
    - choose exactly 2 * num_pairs sentences
    - pair them in sequence
    - sample one valid token from each sentence
    """
    if len(sentence_indices) < 2 * num_pairs:
        raise ValueError("Need at least 2 * num_pairs sentences for paired sampling.")

    rng = random.Random(seed)
    shuffled = list(sentence_indices)
    rng.shuffle(shuffled)
    paired_sentences = shuffled[: 2 * num_pairs]

    pairs: List[AnisotropyPair] = []
    for pair_idx in range(num_pairs):
        sent_a = paired_sentences[2 * pair_idx]
        sent_b = paired_sentences[2 * pair_idx + 1]

        valid_a = list(valid_positions_per_sentence[sent_a])
        valid_b = list(valid_positions_per_sentence[sent_b])
        if not valid_a or not valid_b:
            raise ValueError("Found a sentence with no valid token position.")

        pos_a = rng.choice(valid_a)
        pos_b = rng.choice(valid_b)
        pairs.append((sent_a, pos_a, sent_b, pos_b))

    return pairs


def compute_anisotropy_per_layer(
    vector_lookup: Dict[Tuple[int, int], torch.Tensor],
    pairs: Sequence[AnisotropyPair],
    num_layers: int,
) -> List[float]:
    """Average cosine over sampled token pairs for each layer."""
    if not pairs:
        return [float("nan")] * num_layers

    sums = torch.zeros(num_layers, dtype=torch.float32)
    for sent_a, pos_a, sent_b, pos_b in pairs:
        vec_a = vector_lookup[(sent_a, pos_a)]  # (num_layers, hidden_dim)
        vec_b = vector_lookup[(sent_b, pos_b)]  # (num_layers, hidden_dim)
        sums += F.cosine_similarity(vec_a, vec_b, dim=1)

    return (sums / len(pairs)).tolist()


def select_self_similarity_words(
    word_sentence_sets: Dict[str, set],
    num_words: int,
    min_sentences: int,
    seed: int,
) -> List[str]:
    """Select words appearing in at least `min_sentences` different sentences."""
    eligible_words = [word for word, sentence_set in word_sentence_sets.items() if len(sentence_set) >= min_sentences]
    if len(eligible_words) < num_words:
        raise ValueError(
            f"Not enough eligible words: need {num_words}, found {len(eligible_words)} with >= {min_sentences} sentences."
        )

    rng = random.Random(seed)
    eligible_words = sorted(eligible_words)
    return rng.sample(eligible_words, num_words)


def sample_self_similarity_contexts(
    word_occurrences: Dict[str, List[Tuple[int, int]]],
    selected_words: Sequence[str],
    contexts_per_word: int,
    seed: int,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    For each selected word, sample contextual occurrences from different sentences.
    """
    rng = random.Random(seed)
    sampled_contexts: Dict[str, List[Tuple[int, int]]] = {}

    for word in selected_words:
        sentence_to_positions: Dict[int, List[int]] = defaultdict(list)
        for sentence_idx, token_pos in word_occurrences[word]:
            sentence_to_positions[sentence_idx].append(token_pos)

        sentence_ids = sorted(sentence_to_positions.keys())
        if len(sentence_ids) < contexts_per_word:
            raise ValueError(
                f"Word '{word}' has only {len(sentence_ids)} sentence contexts; need {contexts_per_word}."
            )

        selected_sentence_ids = rng.sample(sentence_ids, contexts_per_word)
        contexts = []
        for sentence_idx in selected_sentence_ids:
            token_pos = rng.choice(sentence_to_positions[sentence_idx])
            contexts.append((sentence_idx, token_pos))
        sampled_contexts[word] = contexts

    return sampled_contexts


def compute_self_similarity_per_layer(
    vector_lookup: Dict[Tuple[int, int], torch.Tensor],
    word_contexts: Dict[str, List[Tuple[int, int]]],
    num_layers: int,
) -> List[float]:
    """
    Compute average within-word contextual similarity per layer.

    For each word:
      - gather vectors for sampled contexts
      - compute average pairwise cosine across contexts for each layer
    Then average across words.
    """
    if not word_contexts:
        return [float("nan")] * num_layers

    layer_sums = torch.zeros(num_layers, dtype=torch.float32)
    word_count = 0

    for contexts in word_contexts.values():
        vectors = torch.stack([vector_lookup[(sent_idx, token_pos)] for sent_idx, token_pos in contexts], dim=0)
        # vectors: (num_contexts, num_layers, hidden_dim)
        num_contexts = vectors.size(0)
        if num_contexts < 2:
            continue

        layer_vectors = vectors.transpose(0, 1)  # (num_layers, num_contexts, hidden_dim)
        normalized = F.normalize(layer_vectors, dim=2)
        similarity_matrices = normalized @ normalized.transpose(1, 2)  # (num_layers, num_contexts, num_contexts)

        tri = torch.triu_indices(num_contexts, num_contexts, offset=1)
        word_layer_scores = similarity_matrices[:, tri[0], tri[1]].mean(dim=1)

        layer_sums += word_layer_scores
        word_count += 1

    if word_count == 0:
        return [float("nan")] * num_layers

    return (layer_sums / word_count).tolist()


def adjusted_self_similarity(self_similarity: Sequence[float], anisotropy: Sequence[float]) -> List[float]:
    """Adjusted self-similarity = self-similarity - anisotropy."""
    return [float(self_similarity[i] - anisotropy[i]) for i in range(len(self_similarity))]
