"""CPU-testable selection rules from the HIVE paper and its appendix."""

import math

import numpy as np


def group_type(rewards):
    """Classify exact all-correct/all-format-only groups, not their mean."""
    values = np.asarray(rewards)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("A reward group must be a nonempty vector")
    # Match the representation of rewards (typically float32) when comparing 0.1.
    if np.all(values == np.asarray(1.0, dtype=values.dtype)):
        return "easy"
    hard_reward = np.asarray(0.1, dtype=values.dtype) if np.issubdtype(values.dtype, np.floating) else 0.1
    if np.all(values == hard_reward):
        return "hard"
    return None


def acceptance_probability(history, entropy_score, p_easy, p_hard,
                           fusion_weight=0.9, reward_floor=0.01):
    """One Bernoulli draw per prompt; unseen prompts receive a free visit."""
    if not history:
        return 1.0
    trailing = 0
    for rewards in reversed(history):
        values = np.asarray(rewards)
        if not np.all(values == values[0]):
            break
        trailing += 1
    kind = group_type(history[-1])
    probability = {"easy": p_easy, "hard": p_hard}.get(kind, 1.0)
    reward_score = max(probability ** trailing, reward_floor)
    if entropy_score is None:
        return reward_score
    return float(np.clip(fusion_weight * reward_score
                         + (1 - fusion_weight) * entropy_score, 0, 1))


def normalize_entropies(entropies, epsilon=1e-8):
    """Min-max normalization over the current pool of latest response entropies."""
    if not entropies:
        return {}
    low, high = min(entropies.values()), max(entropies.values())
    return {key: (value - low) / (high - low + epsilon)
            for key, value in entropies.items()}


def prompt_entropy_scores(entropies, attention_mask):
    """Average H(p(x_l | x_<l)) for l=2..L; logits are source-position aligned."""
    values = np.asarray(entropies)
    mask = np.asarray(attention_mask, dtype=bool)
    if values.shape != mask.shape or values.ndim != 2:
        raise ValueError("Entropy and mask must have the same [batch, length] shape")
    valid = mask[:, :-1] & mask[:, 1:]
    counts = valid.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError("Each prompt must contain at least two valid tokens")
    return np.where(valid, values[:, :-1], 0).sum(axis=1) / counts


def select_prompt_indices(scores, keep_ratio=0.5, offset_ratio=0.0, group_size=8):
    """Rank one raw micro-batch and round DOWN to complete groups of G prompts."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError("Scores must be a finite one-dimensional vector")
    if group_size < 1 or not 0 < keep_ratio <= 1 or not 0 <= offset_ratio < 1:
        raise ValueError("Invalid group size or selection ratios")
    if keep_ratio + offset_ratio > 1:
        raise ValueError("Selection ratios exceed the candidate pool")
    count = math.floor(keep_ratio * len(scores) / group_size) * group_size
    offset = math.floor(offset_ratio * len(scores))
    return np.argsort(-scores, kind="stable")[offset:offset + count].tolist()


def next_candidate_target(missing_groups, zero_variance_ratio, base_target,
                          minimum=64, safety_factor=1.25):
    if missing_groups <= 0:
        return 0
    if not 0 <= zero_variance_ratio <= 1:
        raise ValueError("Zero-variance ratio must lie in [0, 1]")
    if zero_variance_ratio == 1:
        return base_target
    estimate = math.ceil(safety_factor * missing_groups / (1 - zero_variance_ratio))
    return max(minimum, min(base_target, estimate))


def update_exploration(probability, observed_ratio, target, step=0.01,
                       minimum=0.05, maximum=0.95):
    direction = int(target > observed_ratio) - int(target < observed_ratio)
    return float(np.clip(probability + step * direction, minimum, maximum))
