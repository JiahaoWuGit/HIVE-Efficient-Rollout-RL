"""Exercise driver setup and GRPO routing without importing the GPU runtime."""
import ast
from enum import Enum
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from hydra import compose, initialize_config_dir
from load_module import ROOT, load


class TrainerEntryTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse((ROOT / "verl/trainer/ppo/ray_trainer.py").read_text())
        names = {"Role", "AdvantageEstimator", "compute_advantage", "RayPPOTrainer"}
        nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
        self.profiler = Mock()
        self.core = Mock()
        self.namespace = {"Enum": Enum, "DataProto": object, "WorkerType": type,
                          "ResourcePoolManager": object, "RayWorkerGroup": object,
                          "DataProfiler": self.profiler, "ValidationGenerationsLogger": Mock(),
                          "core_algos": self.core}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "trainer_entry", "exec"), self.namespace)

    def test_main_driver_initializes_without_removed_roles_or_configs(self):
        launcher = load("scripts/train.py")
        args = launcher.parser().parse_args(["--output-dir", "/tmp/hive-entry-test"])
        command, _, _, _ = launcher.build_command(args)
        with initialize_config_dir(version_base=None, config_dir=str(ROOT / "verl/trainer/config")):
            config = compose(config_name="ppo_trainer", overrides=command[4:])
        trainer_cls = self.namespace["RayPPOTrainer"]
        # The actual config validation runs; dataset/GPU loading is outside this CPU test.
        trainer_cls._create_dataloader = Mock()
        trainer = trainer_cls(config=config, tokenizer=object(),
                              role_worker_mapping={self.namespace["Role"].ActorRollout: object},
                              resource_pool_manager=object())
        self.assertEqual(trainer.p_easy, 0.5)
        self.assertEqual(trainer.p_hard, 0.5)
        self.profiler.assert_called_once_with(fusion_weight=0.9, reward_floor=0.01, seed=42)
        self.assertNotIn("critic", config)
        self.assertNotIn("ref", config.actor_rollout_ref)

    def test_advantage_route_retains_response_mask_and_group_ids(self):
        outputs = (np.ones((8, 3)), np.ones((8, 3)), np.arange(8))
        self.core.compute_grpo_outcome_advantage.return_value = outputs
        batch = SimpleNamespace(batch={"responses": SimpleNamespace(size=lambda dim: 3),
                                       "attention_mask": np.ones((8, 5)),
                                       "token_level_rewards": np.zeros((8, 3))},
                                non_tensor_batch={"uid": np.array(["prompt-a"] * 8)})
        compute = self.namespace["compute_advantage"]
        self.assertIs(compute(batch, "grpo"), batch)
        kwargs = self.core.compute_grpo_outcome_advantage.call_args.kwargs
        self.assertEqual(kwargs["eos_mask"].shape, (8, 3))
        np.testing.assert_array_equal(kwargs["index"], ["prompt-a"] * 8)
        np.testing.assert_array_equal(batch.batch["reward"], outputs[2])
        with self.assertRaises(ValueError):
            compute(batch, "gae")
