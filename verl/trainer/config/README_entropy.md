# HIVE selector configuration

The main-paper entry point is `python scripts/train.py`.

| Option | HIVE default | Meaning |
| --- | ---: | --- |
| `data.history_fusion_weight` | 0.9 | Weight on reward history |
| `data.history_reward_floor` | 0.01 | Floor on the reward-history score |
| `data.entropy_keep_ratio` | 0.5 | Keep the highest-scoring half of each raw micro-batch |
| `data.entropy_start_ratio` | 0.0 | Do not trim the highest-entropy candidates |
| `data.target_zero_variance` | 0.25 | Total easy/hard target, split 1:2 |
| `data.resample_minimum` | 64 | Lower bound on the post-gate top-up target |
| `data.resample_safety_factor` | 1.25 | Top-up safety factor |

Prompt scores exclude padding and the final source position, averaging the
next-token entropies for prompt tokens 2 through L. Selection is stable on ties
and the kept prompt count is rounded down to a multiple of G=8. Empty results
skip to the next raw micro-batch. Candidates accumulate **after** this gate.

History stores complete reward groups and the latest mean response entropy.
The entropy prior uses min-max normalization across the current metadata pool.
History and exploration probabilities update once after each GRPO update,
including discarded and excess generated groups.
