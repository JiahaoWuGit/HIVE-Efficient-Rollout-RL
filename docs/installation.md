# Installation

Use a dedicated Linux environment with Python 3.11 and NVIDIA CUDA. The paper uses 8× A100 for the main study. The 4-GPU wrapper is a convenience configuration; it is not a replacement for the paper's runtime setup.

```bash
conda env create -f environment.yml
conda activate hive
python -m pip install -r requirements.txt
python -m pip install ninja
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip install -e . --no-deps
python -m pip check
python scripts/check_environment.py
```

`requirements.txt` retains the repository's pinned training stack: PyTorch 2.6.0, vLLM 0.8.2, Transformers 4.49.0, TensorDict 0.6.0, and Ray 2.43.0. FlashAttention is installed after PyTorch because its build imports PyTorch. Use a CUDA 12.4 toolkit and a compatible NVIDIA driver for this stack; a source build also needs a C++ compiler and `ninja`. See the upstream [FlashAttention installation instructions](https://github.com/Dao-AILab/flash-attention#installation-and-features) and [PyTorch 2.6 packages](https://pytorch.org/get-started/previous-versions/#v260).

The launcher defaults to `VLLM_ATTENTION_BACKEND=XFORMERS`, tensor parallel size 1, and FSDP training. Reduce `--micro-batch` or `--log-prob-micro-batch` if memory is insufficient; this preserves the final training batch. Tensor parallelism must divide `--gpus`.

## Model compatibility

Qwen2.5-Math is the default entry point. DeepSeek-R1-Distill-Qwen and Llama use the same launch interface with the paper's model-specific context and batch sizes. Llama access requires the model provider's authorization and Hugging Face authentication.

The Qwen3-4B-Base profile deliberately requires `--model-path`: use the exact base checkpoint from your experiment. The pinned Transformers 4.49 / vLLM 0.8.2 stack predates Qwen3 support; see the [official Qwen3 runtime requirements](https://github.com/QwenLM/Qwen3#quickstart). A native Qwen3 checkpoint needs a compatible Transformers/vLLM environment and corresponding verl integration. The profile records the paper settings; it does not assert that this legacy dependency set supports that architecture. Preserve the working server environment and run the integration check below before a full run.

Do not replace the vendored `verl/` directory by installing a newer upstream verl package over it: this directory contains HIVE's trainer integration.

## Verify on the training server

```bash
python scripts/check_environment.py
python scripts/train.py --dry-run
python scripts/train.py --steps 5 --test-freq 5 --save-freq 5 --output-dir outputs/smoke
python scripts/summarize_results.py outputs/smoke/metrics.jsonl
```

The short run checks GPU execution, scoring, validation, and checkpoint writing. It is not a reproduction of the paper's accuracy or speedup. CPU CI checks selection rules, data conversion, launcher configurations, Python syntax, and shell syntax; it does not run CUDA kernels or distributed training.

For a CPU-only development check, install `numpy==1.26.4 hydra-core==1.3.2 pyflakes==3.4.0`, then run:

```bash
python -m unittest discover -s tests -v
python scripts/check_repository.py
```
