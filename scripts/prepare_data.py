#!/usr/bin/env python3
"""Prepare the main math datasets using the archived HIVE/GRESO mappings."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples/data_preprocess"))
from math_answers import last_boxed_only_string, remove_boxed

GAOKAO_URL = ("https://raw.githubusercontent.com/Infini-AI-Lab/GRESO/"
              "b27784315479921ea1977178d7e6c4858b2ae5a1/raw-data/gaokao-2023en-test.jsonl")
GAOKAO_SHA256 = "e1503ee5bd33bacb677b707344d64cfa44c9b841428544a14df7a6b643376977"
# name: dataset ID, source split, problem field, answer field, output split, limit
SPECS = {
    "dapo_math": ("haizhongzheng/DAPO-Math-17K-cleaned", "train", "prompt", "target", "train", None),
    "lighteval-math": ("xDAN2099/lighteval-MATH", "train", "problem", "solution", "train", 7500),
    "openr1_math": ("open-r1/OpenR1-Math-220k", "train", "problem", "answer", "train", 30000),
    "math500": ("HuggingFaceH4/MATH-500", "test", "problem", "answer", "test", None),
    "aime2024": ("HuggingFaceH4/aime_2024", "train", "problem", "answer", "test", None),
    "amc": ("AI-MO/aimo-validation-amc", "train", "problem", "answer", "test", None),
    "minervamath": ("math-ai/minervamath", "test", "question", "answer", "test", None),
    "olympiadbench": ("math-ai/olympiadbench", "test", "question", "final_answer", "test", None),
    "gaokao": (GAOKAO_URL, "test", "question", "answer", "test", None),
}
EVALUATION = ("math500", "aime2024", "amc", "gaokao", "minervamath", "olympiadbench")


def math_prompt(problem):
    return (f"Please solve the following math problem: {problem}. "
            "The assistant first thinks about the reasoning process step by step "
            "and then provides the user with the answer. Return the final answer "
            "in \\boxed{} tags, for example \\boxed{1}. Let's solve this step by step. ")


def convert_row(name, example, index):
    source, _, problem_key, answer_key, split, _ = SPECS[name]
    answer = example[answer_key]
    if name == "lighteval-math":
        boxed = last_boxed_only_string(answer)
        if boxed is None:
            raise ValueError(f"MATH example {index} has no boxed ground truth")
        answer = remove_boxed(boxed)
    elif name == "olympiadbench":
        answer = answer[0].replace("$", "")
    elif name == "gaokao":
        answer = answer.replace("$", "")
    if isinstance(answer, float) and answer.is_integer():
        answer = int(answer)
    data_source = name if name in ("dapo_math", "gaokao") else source
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": math_prompt(example[problem_key])}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(answer)},
        "extra_info": {"split": split, "index": index},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("dapo-math", "openr1", "eval", "all"), default="dapo-math")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--gaokao-jsonl", type=Path, help="Use a local copy of the documented Gaokao input")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--revisions-file", type=Path, help="JSON mapping of dataset IDs to source commits")
    args = parser.parse_args()
    from datasets import Dataset, load_dataset
    from huggingface_hub import HfApi
    revisions = json.loads(args.revisions_file.read_text()) if args.revisions_file else {}
    api = HfApi()
    names = list(EVALUATION)
    if args.dataset in ("dapo-math", "all"):
        names = ["dapo_math", "lighteval-math"] + names
    if args.dataset in ("openr1", "all"):
        names = ["openr1_math"] + names
    for name in names:
        source, source_split, _, _, split, limit = SPECS[name]
        target = args.output_dir / name / f"{split}.parquet"
        if target.exists() and not args.overwrite:
            print(f"Exists: {target} (use --overwrite to replace)")
            continue
        revision = None
        if name == "gaokao":
            raw = args.gaokao_jsonl.read_bytes() if args.gaokao_jsonl else urllib.request.urlopen(GAOKAO_URL, timeout=60).read()
            if hashlib.sha256(raw).hexdigest() != GAOKAO_SHA256:
                raise ValueError("Gaokao input does not match the archived experiment input")
            dataset = Dataset.from_list([json.loads(line) for line in raw.decode().splitlines() if line.strip()])
        else:
            revision = revisions.get(source) or api.dataset_info(source).sha
            dataset = load_dataset(source, split=source_split, revision=revision, trust_remote_code=True)
        if limit is not None:
            if len(dataset) < limit:
                raise ValueError(f"{source} has fewer than {limit} rows")
            dataset = dataset.select(range(limit))
        source_fingerprint = dataset._fingerprint
        prepared = dataset.map(lambda row, index: convert_row(name, row, index),
                               with_indices=True, remove_columns=dataset.column_names)
        target.parent.mkdir(parents=True, exist_ok=True)
        prepared.to_parquet(str(target))
        metadata = {"source": source, "source_split": source_split, "revision": revision,
                    "source_fingerprint": source_fingerprint, "rows": len(prepared),
                    "selection": f"first {limit} rows" if limit else "complete split",
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        target.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Prepared {len(prepared)} rows: {target}")


if __name__ == "__main__":
    main()
