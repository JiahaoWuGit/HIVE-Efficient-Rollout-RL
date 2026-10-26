#!/usr/bin/env python3
"""Select the best six-benchmark validation checkpoint from trainer JSONL metrics."""

import argparse
import json
import math
from pathlib import Path

PREFIX = "val/test_score/"
BENCHMARKS = (
    "HuggingFaceH4/MATH-500", "HuggingFaceH4/aime_2024", "AI-MO/aimo-validation-amc",
    "gaokao", "math-ai/minervamath", "math-ai/olympiadbench",
)


def best_checkpoint(records):
    # On resume, a repeated step supersedes the previous record at that step.
    by_step = {row["step"]: row for row in records}
    candidates = []
    for row in by_step.values():
        scores = {name: row[PREFIX + name] for name in BENCHMARKS if PREFIX + name in row}
        if len(scores) != len(BENCHMARKS):
            continue
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in scores.values()):
            raise ValueError(f"Invalid validation accuracy at step {row['step']}")
        average = sum(scores.values()) / len(scores)
        candidates.append((average, -row["step"], row, scores))
    if not candidates:
        raise ValueError("No complete six-benchmark validation record found")
    average, _, row, scores = max(candidates, key=lambda item: item[:2])
    return {"step": row["step"], "average_accuracy_percent": round(100 * average, 4),
            "benchmark_accuracy_percent": {name: round(100 * score, 4) for name, score in scores.items()},
            "generated_responses": row.get("sampling/sampling_total_num"),
            "checkpoint": f"global_step_{row['step']}" if row["step"] > 0 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.metrics.read_text().splitlines() if line.strip()]
    try:
        result = best_checkpoint(records)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
