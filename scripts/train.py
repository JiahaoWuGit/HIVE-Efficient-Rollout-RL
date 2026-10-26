#!/usr/bin/env python3
"""Launch the paper's main math experiments; --dry-run needs only Python."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = json.loads((ROOT / "configs/models.json").read_text())
EVAL_SETS = ("math500", "aime2024", "amc", "gaokao", "minervamath", "olympiadbench")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODELS, default="qwen25-math-1.5b")
    p.add_argument("--model-path", default=os.environ.get("MODEL_PATH"))
    p.add_argument("--dataset", choices=("dapo-math", "openr1"), default="dapo-math")
    p.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT", ROOT / "data")))
    p.add_argument("--gpus", type=int, choices=(1, 2, 4, 8), default=int(os.environ.get("GPU_NUM", 8)))
    p.add_argument("--steps", type=int, default=int(os.environ.get("MAX_STEPS", 1000)))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-freq", type=int, default=int(os.environ.get("SAVE_FREQ", 50)))
    p.add_argument("--test-freq", type=int, default=int(os.environ.get("TEST_FREQ", 50)))
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--log-prob-micro-batch", type=int, default=8)
    p.add_argument("--tensor-parallel", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    p.add_argument("--project", default="hive")
    p.add_argument("--name")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--wandb", action="store_true", help="Enable W&B after running wandb login")
    p.add_argument("--resume", action="store_true", help="Resume a new-format run in --output-dir")
    p.add_argument("--eval-only", action="store_true", help="Evaluate --model-path without training")
    p.add_argument("--dry-run", action="store_true", help="Print resolved command without loading dependencies or data")
    return p


def build_command(args):
    profile = MODELS[args.model]
    model_path = args.model_path or profile["model_path"]
    if not model_path:
        raise ValueError("Pass --model-path for the Qwen3-4B-Base checkpoint used in your experiment.")
    if args.steps < 1 or args.micro_batch < 1 or args.log_prob_micro_batch < 1:
        raise ValueError("Steps and micro-batch sizes must be positive")
    if args.tensor_parallel < 1 or args.gpus % args.tensor_parallel:
        raise ValueError("Tensor parallelism must divide the GPU count")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("GPU memory utilization must lie in (0, 1)")
    if (64 * 8 // args.gpus) % args.micro_batch:
        raise ValueError("Actor micro-batch must divide the 512-response mini-batch per GPU")
    if args.resume and args.output_dir is None:
        raise ValueError("--resume requires --output-dir")
    root = args.data_root.expanduser().resolve()
    train = ([root / "dapo_math/train.parquet", root / "lighteval-math/train.parquet"]
             if args.dataset == "dapo-math" else [root / "openr1_math/train.parquet"])
    validation = [root / name / "test.parquet" for name in EVAL_SETS]
    name = args.name or f"{args.model}-{args.dataset}-hive-seed{args.seed}"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output_dir or ROOT / "outputs" / name / timestamp).expanduser().resolve()
    # ++ permits explicit paper settings absent from the upstream generic config.
    config = {
        "algorithm.adv_estimator": "grpo",
        "data.train_files": [str(p) for p in train],
        "data.val_files": [str(p) for p in validation],
        "data.seed": args.seed,
        "data.train_batch_size": 32,
        "data.real_train_batch_size": profile["train_prompts"],
        "data.sampling_batch_size": profile["candidate_prompts"],
        "data.history_fusion_weight": 0.9,
        "data.history_reward_floor": 0.01,
        "data.p_easy": 0.5, "data.p_hard": 0.5,
        "data.target_zero_variance": 0.25,
        "data.exploration_step": 0.01,
        "data.min_p": 0.05, "data.max_p": 0.95,
        "data.entropy_keep_ratio": 0.5, "data.entropy_start_ratio": 0.0,
        "data.resample_minimum": 64, "data.resample_safety_factor": 1.25,
        "data.max_prompt_length": 1536,
        "data.max_response_length": profile["context_length"] - 1536,
        "data.truncation": "left",
        "data.val_batch_size": 8,
        "actor_rollout_ref.model.path": model_path,
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.actor.optim.lr": 1e-6,
        "actor_rollout_ref.actor.optim.betas": [0.9, 0.999],
        "actor_rollout_ref.actor.optim.weight_decay": 0.01,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 64,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": args.micro_batch,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.fsdp_config.param_offload": False,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": False,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.n": 8,
        "actor_rollout_ref.rollout.seed": args.seed,
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": args.tensor_parallel,
        "actor_rollout_ref.rollout.gpu_memory_utilization": args.gpu_memory_utilization,
        "actor_rollout_ref.rollout.max_num_batched_tokens": max(8192, profile["context_length"]),
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": args.log_prob_micro_batch,
        "actor_rollout_ref.rollout.val_kwargs.n": 4,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": True,
        "actor_rollout_ref.rollout.val_kwargs.top_p": 0.7,
        "actor_rollout_ref.rollout.val_kwargs.temperature": 1.0,
        "trainer.n_gpus_per_node": args.gpus, "trainer.nnodes": 1,
        "trainer.max_steps": args.steps, "trainer.total_training_steps": args.steps,
        "trainer.total_epochs": 2000,
        "trainer.save_freq": args.save_freq, "trainer.test_freq": args.test_freq,
        "trainer.val_before_train": args.eval_only, "trainer.val_only": args.eval_only,
        "trainer.logger": ["console", "wandb"] if args.wandb else ["console"],
        "trainer.project_name": args.project, "trainer.experiment_name": name,
        "trainer.metrics_path": str(output / "metrics.jsonl"),
        "trainer.default_local_dir": str(output / "checkpoints"),
        "trainer.resume_mode": "auto" if args.resume else "disable",
        "trainer.keep_all_ckpts": True,
    }
    command = [sys.executable, "-u", "-m", "verl.trainer.main_ppo"]
    command += [f"++{key}={json.dumps(value)}" for key, value in config.items()]
    return command, config, output, train + validation


def main():
    args = parser().parse_args()
    try:
        command, config, output, inputs = build_command(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.dry_run:
        print(shlex.join(command))
        return
    missing = [str(p) for p in inputs if not p.is_file()]
    if missing:
        raise SystemExit("Prepare the data first (see docs/data.md):\n" + "\n".join(missing))
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise SystemExit("Output directory is not empty; use a new directory or --resume.")
    config_file = output / "resolved-config.json"
    if args.resume:
        if not config_file.is_file():
            raise SystemExit("Resume requires an existing run with resolved-config.json.")
        previous = json.loads(config_file.read_text())
        allowed = {"trainer.resume_mode", "trainer.max_steps", "trainer.total_training_steps", "trainer.logger",
                   "trainer.save_freq", "trainer.test_freq"}
        changed = [key for key, value in config.items() if key not in allowed and previous.get(key) != value]
        if changed:
            raise SystemExit("Resume would change the experiment configuration: " + ", ".join(changed))
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved-config.json").write_text(json.dumps(config, indent=2) + "\n")
    env = os.environ.copy()
    env.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
    env.setdefault("RAY_TMPDIR", "/tmp/ray-hive")
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    # Ray is owned by the training process; never stop another user's cluster.
    with (output / "train.log").open("a") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            status = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
    raise SystemExit(status)


if __name__ == "__main__":
    main()
