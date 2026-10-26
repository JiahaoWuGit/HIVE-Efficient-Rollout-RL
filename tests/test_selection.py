import ast
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from load_module import ROOT, load

rules = load("verl/trainer/ttis/hive_selection.py")


class SelectionTests(unittest.TestCase):
    def test_group_classification_uses_complete_rewards(self):
        self.assertEqual(rules.group_type(np.full(8, 0.1, dtype=np.float32)), "hard")
        self.assertEqual(rules.group_type([1] * 8), "easy")
        self.assertIsNone(rules.group_type([0, 0.2] * 4))
        self.assertIsNone(rules.group_type([0] * 8))
        with self.assertRaises(ValueError):
            rules.group_type([])

    def test_unseen_free_visit_and_single_mixture(self):
        self.assertEqual(rules.acceptance_probability([], 0, 0.5, 0.5), 1)
        self.assertAlmostEqual(rules.acceptance_probability([[1]*8], 0.2, 0.5, 0.5), 0.47)
        self.assertAlmostEqual(rules.acceptance_probability([[1]*8], None, 0.5, 0.5), 0.5)

    def test_trailing_run_reset_and_floor(self):
        varied = [0.1, 1.0] * 4
        history = [[1]*8, varied, [0.1]*8, [1]*8]
        self.assertAlmostEqual(rules.acceptance_probability(history, None, 0.5, 0.7), 0.25)
        self.assertEqual(rules.acceptance_probability(history + [varied], None, 0.5, 0.7), 1)
        self.assertEqual(rules.acceptance_probability([[1]*8]*100, None, 0.05, 0.5), 0.01)
        self.assertEqual(rules.acceptance_probability([[0]*8], None, 0.5, 0.5), 1)

    def test_minmax_over_latest_pool(self):
        self.assertEqual(rules.normalize_entropies({}), {})
        self.assertEqual(rules.normalize_entropies({"x": 4, "y": 4}), {"x": 0, "y": 0})
        result = rules.normalize_entropies({"x": 1, "y": 2, "z": 5})
        self.assertAlmostEqual(result["y"], 0.25)
        self.assertAlmostEqual(result["z"], 1)

    def test_prompt_mask_alignment_for_both_padding_sides(self):
        entropy = [[1000, 2, 4, 999], [2, 4, 999, 1000]]
        mask = [[0, 1, 1, 1], [1, 1, 1, 0]]
        np.testing.assert_allclose(rules.prompt_entropy_scores(entropy, mask), [3, 3])
        with self.assertRaises(ValueError):
            rules.prompt_entropy_scores([[1, 9]], [[0, 1]])
        with self.assertRaises(ValueError):
            rules.prompt_entropy_scores([[1, 9]], [[1]])

    def test_rounding_and_stable_ties(self):
        self.assertEqual(rules.select_prompt_indices(list(range(32))), list(range(31, 15, -1)))
        self.assertEqual(len(rules.select_prompt_indices(range(31))), 8)
        self.assertEqual(rules.select_prompt_indices(range(15)), [])
        self.assertEqual(rules.select_prompt_indices([5]*32), list(range(16)))
        with self.assertRaises(ValueError):
            rules.select_prompt_indices([np.nan])
        with self.assertRaises(ValueError):
            rules.select_prompt_indices(range(32), keep_ratio=0.8, offset_ratio=0.3)

    def test_topup_ceil_bounds_and_all_zero_variance(self):
        self.assertEqual(rules.next_candidate_target(65, 0, 384), 82)
        self.assertEqual(rules.next_candidate_target(1, 0, 384), 64)
        self.assertEqual(rules.next_candidate_target(250, 0.9, 384), 384)
        self.assertEqual(rules.next_candidate_target(100, 1, 192), 192)
        self.assertEqual(rules.next_candidate_target(0, 1, 192), 0)

    def test_probability_update_direction_equality_and_clipping(self):
        self.assertAlmostEqual(rules.update_exploration(0.5, 0.2, 0.25), 0.51)
        self.assertAlmostEqual(rules.update_exploration(0.5, 0.3, 0.25), 0.49)
        self.assertEqual(rules.update_exploration(0.5, np.float64(0.25), 0.25), 0.5)
        self.assertEqual(rules.update_exploration(0.95, 0, 0.25), 0.95)
        self.assertEqual(rules.update_exploration(0.05, 1, 0.25), 0.05)


class HistoryIntegrationTests(unittest.TestCase):
    """Exercise the actual profiler class with only its torch I/O/RNG substituted."""
    def setUp(self):
        def save(state, path):
            Path(path).write_bytes(pickle.dumps(state))
        class Generator:
            def manual_seed(self, seed):
                self.state = seed
                return self
            def get_state(self):
                return self.state
            def set_state(self, state):
                self.state = state
        self.torch = SimpleNamespace(rand=Mock(return_value=np.array([0.46])), save=save,
                                     Generator=Generator,
                                     load=lambda path, **kwargs: pickle.loads(Path(path).read_bytes()))
        tree = ast.parse((ROOT / "verl/trainer/ttis/presampling_selector.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DataProfiler")
        namespace = {"torch": self.torch, "acceptance_probability": rules.acceptance_probability,
                     "group_type": rules.group_type}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "profiler", "exec"), namespace)
        self.Profiler = namespace["DataProfiler"]
        self.profiler = self.Profiler()

    def test_one_draw_per_seen_prompt_and_unseen_bypass(self):
        self.profiler.add_reward_group(0, "math_1", np.ones(8))
        self.profiler.set_entropy_scores({"math_1": 0.2})
        batch = SimpleNamespace(non_tensor_batch={"data_source": np.array(["math", "math"]),
                                                 "index": np.array([1, 2])},
                                select_via_index=lambda indices: indices)
        self.assertEqual(self.profiler.filter_examples_all_probabilistic(1, batch, 0.5, 0.5), [0, 1])
        self.assertEqual(self.torch.rand.call_count, 1)
        self.torch.rand.return_value = np.array([0.48])
        self.assertEqual(self.profiler.filter_examples_all_probabilistic(1, batch, 0.5, 0.5), [1])

    def test_roundtrip_preserves_reward_groups_and_rejects_old_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.pt"
            rewards = np.full(8, 0.1, dtype=np.float32)
            self.profiler.add_reward_group(2, "math_5", rewards)
            self.profiler.set_entropy_scores({"math_5": 0.75})
            self.profiler.generator.set_state(125)
            self.profiler.save(path)
            restored = self.Profiler()
            restored.load(path)
            np.testing.assert_array_equal(restored.reward_groups["math_5"][0], rewards)
            self.assertEqual(restored.entropy_scores["math_5"], 0.75)
            self.assertEqual(restored.generator.get_state(), 125)
            self.torch.save({"data": {}}, path)
            with self.assertRaisesRegex(ValueError, "full reward-group history"):
                restored.load(path)
