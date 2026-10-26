#!/usr/bin/env python3
"""Check imports and CUDA availability before a training run."""

from importlib import import_module, metadata
import platform
import sys


def main():
    failed = []
    print(f"Python: {platform.python_version()} on {platform.system()}")
    if sys.version_info[:2] != (3, 11):
        failed.append("Use Python 3.11 for the pinned training environment")
    if platform.system() != "Linux":
        failed.append("Distributed training requires the documented Linux/CUDA environment")
    for package, module in (("torch", "torch"), ("vllm", "vllm"),
                            ("transformers", "transformers"), ("ray", "ray"),
                            ("tensordict", "tensordict"), ("flash-attn", "flash_attn"),
                            ("datasets", "datasets"), ("hydra-core", "hydra")):
        try:
            import_module(module)
            print(f"{package}: {metadata.version(package)}")
        except Exception as error:
            failed.append(f"{package}: {error}")
    try:
        import torch
        print(f"Torch CUDA: {torch.version.cuda}; visible GPUs: {torch.cuda.device_count()}")
        if not torch.cuda.is_available():
            failed.append("CUDA is not available to PyTorch")
        for index in range(torch.cuda.device_count()):
            print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
        import verl
        print(f"verl source: {verl.__file__}")
    except Exception as error:
        failed.append(str(error))
    if failed:
        print("\nEnvironment checks failed:", file=sys.stderr)
        for message in failed:
            print(f"- {message}", file=sys.stderr)
        raise SystemExit(1)
    print("Imports and CUDA checks passed. Next: the five-step integration run in docs/installation.md.")


if __name__ == "__main__":
    main()
