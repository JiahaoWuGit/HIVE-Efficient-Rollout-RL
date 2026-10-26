# Data preparation

The launcher consumes parquet files with `data_source`, a chat-format `prompt`, `ability`, `reward_model`, and `extra_info.index`. Indices are stable within a dataset split and become part of the history key.

```bash
python scripts/prepare_data.py --dataset dapo-math
python scripts/prepare_data.py --dataset openr1
```

Each command also prepares the six evaluation sets. Use `--dataset eval` for evaluation only, `--dataset all` for both training corpora, or `--output-dir /path/to/data` to change the destination. Existing parquet files are skipped; `--overwrite` explicitly regenerates them.

| Output directory | Source | Source split / selection |
| :--- | :--- | :--- |
| `dapo_math` | [DAPO-Math-17K-cleaned](https://huggingface.co/datasets/haizhongzheng/DAPO-Math-17K-cleaned) | train, all rows |
| `lighteval-math` | [lighteval-MATH](https://huggingface.co/datasets/xDAN2099/lighteval-MATH) | train, first 7,500 rows |
| `openr1_math` | [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | train, first 30,000 rows |
| `math500` | [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | test |
| `aime2024` | [AIME 2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024) | train |
| `amc` | [AMC validation](https://huggingface.co/datasets/AI-MO/aimo-validation-amc) | train |
| `gaokao` | [GRESO's Gaokao input](https://github.com/Infini-AI-Lab/GRESO/blob/b27784315479921ea1977178d7e6c4858b2ae5a1/raw-data/gaokao-2023en-test.jsonl) | complete file |
| `minervamath` | [Minerva Math](https://huggingface.co/datasets/math-ai/minervamath) | test |
| `olympiadbench` | [Olympiad Bench](https://huggingface.co/datasets/math-ai/olympiadbench) | test |

Training outputs are `train.parquet`; evaluation outputs are `test.parquet`. Each file has an adjacent `.metadata.json` recording its source, revision, selected rows, dataset fingerprint, and parquet SHA-256. The Gaokao source is pinned by commit and checksum. For offline preparation, pass `--gaokao-jsonl /path/to/gaokao-2023en-test.jsonl` with the same source file.

The converter follows the mappings in the archived HIVE preprocessing scripts. It extracts boxed MATH answers, unwraps Olympiad answers, and applies the paper's math prompt template. Dataset downloads and model weights are not committed to this repository.

## Reproducing a dataset snapshot

For an exact rerun, use the original experiment parquet files and verify their hashes. The historical Hugging Face revisions are not recorded in the source archive, so a fresh download must not be treated as proof of an identical training set. By default, preparation resolves the current dataset commit before loading it and records that revision. To reuse known revisions, provide a JSON mapping from dataset IDs to commit hashes:

```bash
python scripts/prepare_data.py --dataset dapo-math --revisions-file revisions.json
```

Keep the metadata files with experiment logs. Dataset order matters for the 30k Open-R1 subset. Hugging Face downloads may require authentication, network access, or acceptance of the provider's terms.
