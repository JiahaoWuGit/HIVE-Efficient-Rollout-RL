import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from load_module import ROOT, load

launcher = load("scripts/train.py")
data = load("scripts/prepare_data.py")
summary = load("scripts/summarize_results.py")


class LauncherTests(unittest.TestCase):
    def args(self, *values):
        with patch.dict("os.environ", {}, clear=True):
            return launcher.parser().parse_args(values)

    def test_every_main_configuration_composes_with_hydra(self):
        for model in launcher.MODELS:
            for dataset in ("dapo-math", "openr1"):
                with self.subTest(model=model, dataset=dataset):
                    args = self.args("--model", model, "--dataset", dataset,
                                     "--model-path", "/tmp/checkpoint with spaces", "--output-dir", "/tmp/hive-test")
                    command, settings, _, paths = launcher.build_command(args)
                    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "verl/trainer/config")):
                        cfg = compose(config_name="ppo_trainer", overrides=command[4:])
                    for key, value in settings.items():
                        self.assertEqual(OmegaConf.select(cfg, key), value, key)
                    self.assertEqual(cfg.data.train_batch_size, 32)
                    self.assertEqual(cfg.actor_rollout_ref.actor.ppo_mini_batch_size * cfg.actor_rollout_ref.rollout.n, 512)
                    self.assertEqual(cfg.actor_rollout_ref.rollout.response_length + cfg.data.max_prompt_length,
                                     launcher.MODELS[model]["context_length"])
                    expected = (128, 192) if model.startswith("deepseek") else (256, 384)
                    self.assertEqual((cfg.data.real_train_batch_size, cfg.data.sampling_batch_size), expected)
                    self.assertEqual(len(paths), 8 if dataset == "dapo-math" else 7)


    def test_gpu_variant_preserves_update_size(self):
        _, a, _, _ = launcher.build_command(self.args("--gpus", "4"))
        _, b, _, _ = launcher.build_command(self.args("--gpus", "8"))
        self.assertEqual(a["data.real_train_batch_size"], b["data.real_train_batch_size"])
        self.assertEqual(a["data.sampling_batch_size"], b["data.sampling_batch_size"])

    def test_invalid_launch_settings_fail_before_gpu_imports(self):
        for flags in (("--micro-batch", "7"), ("--steps", "0"), ("--resume",),
                      ("--tensor-parallel", "3"), ("--model", "qwen3-4b-base")):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                launcher.build_command(self.args(*flags))

    def test_evaluation_and_seed_override(self):
        _, config, _, _ = launcher.build_command(self.args("--eval-only", "--seed", "7"))
        self.assertTrue(config["trainer.val_before_train"])
        self.assertTrue(config["trainer.val_only"])
        self.assertEqual(config["data.seed"], 7)
        self.assertEqual(config["actor_rollout_ref.rollout.seed"], 7)


class DataTests(unittest.TestCase):
    def test_nested_boxed_answer_and_stable_index(self):
        row = data.convert_row("lighteval-math", {"problem": "Find x", "solution": r"Thus \boxed{\frac{1}{2}}."}, 15)
        self.assertEqual(row["reward_model"]["ground_truth"], r"\frac{1}{2}")
        self.assertEqual(row["extra_info"]["index"], 15)
        self.assertIn(r"\boxed{} tags", row["prompt"][0]["content"])
        with self.assertRaises(ValueError):
            data.convert_row("lighteval-math", {"problem": "x", "solution": "no answer"}, 0)

    def test_scalar_and_list_answers(self):
        row = data.convert_row("amc", {"problem": "x", "answer": 15.0}, 0)
        self.assertEqual(row["reward_model"]["ground_truth"], "15")
        row = data.convert_row("olympiadbench", {"question": "x", "final_answer": ["$42$"]}, 0)
        self.assertEqual(row["reward_model"]["ground_truth"], "42")
        self.assertEqual(tuple(launcher.EVAL_SETS), tuple(data.EVALUATION))


class ReportingTests(unittest.TestCase):
    def test_best_checkpoint_uses_six_dataset_macro_average(self):
        def row(step, values, responses):
            return {"step": step, **{summary.PREFIX + key: value for key, value in zip(summary.BENCHMARKS, values)},
                    "sampling/sampling_total_num": responses}
        # Averaging partial rewards or taking the last checkpoint would choose incorrectly.
        records = [row(50, [0.6]*6, 1000), row(100, [0.5]*6, 2000), {"step": 150, summary.PREFIX + "average": 0.9}]
        result = summary.best_checkpoint(records)
        self.assertEqual(result["step"], 50)
        self.assertEqual(result["average_accuracy_percent"], 60)
        self.assertEqual(result["generated_responses"], 1000)
        records.append(row(50, [0.4]*6, 1100))
        self.assertEqual(summary.best_checkpoint(records)["step"], 100)
        with self.assertRaises(ValueError):
            summary.best_checkpoint([])
