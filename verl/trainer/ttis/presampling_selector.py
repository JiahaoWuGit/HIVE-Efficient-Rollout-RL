import torch

from verl.trainer.ttis.hive_selection import acceptance_probability, group_type

class DataProfiler:
    def __init__(self, fusion_weight=0.9, reward_floor=0.01, seed=42):
        self.fusion_weight = fusion_weight
        self.reward_floor = reward_floor
        self.reward_groups = {}
        self.data = {}
        self.entropy_scores = {}
        self.generator = torch.Generator().manual_seed(seed)

    def set_entropy_scores(self, entropy_scores: dict):
        """
        Set the entropy scores for prompts.
        Args:
            entropy_scores: dict mapping prompt_id (data_source_index) to normalized entropy score
        """
        self.entropy_scores = entropy_scores


    def add_reward(self, epoch: int, data_id: str, reward: float):
        if data_id not in self.data:
            self.data[data_id] = []
        self.data[data_id].append((epoch, reward))

    def get_data_id_list(self, data_sources: list, indices: list):
        data_ids = []
        for data_source, index in zip(data_sources, indices):
            data_ids.append(f"{data_source}_{index}")
        return data_ids

    def filter_examples_all_probabilistic(self, current_epoch, batch, p_easy, p_hard, return_log=False):
        data_ids = self.get_data_id_list(
            batch.non_tensor_batch['data_source'].tolist(),
            batch.non_tensor_batch['index'].astype(int).tolist(),
        )
        selected, skip_log = [], []
        for index, data_id in enumerate(data_ids):
            history = self.reward_groups.get(data_id, [])
            probability = acceptance_probability(
                history, self.entropy_scores.get(data_id), p_easy, p_hard,
                self.fusion_weight, self.reward_floor,
            )
            if not history or torch.rand(1, generator=self.generator).item() < probability:
                selected.append(index)
                skip_log.append([data_id, 2])
            else:
                skip_log.append([data_id, 0 if group_type(history[-1]) == 'easy' else 1])
        result = batch.select_via_index(selected)
        return (result, skip_log) if return_log else result

    def add_reward_group(self, epoch, data_id, rewards):
        self.reward_groups.setdefault(data_id, []).append(rewards)
        self.add_reward(epoch, data_id, float(rewards.mean()))

    def save(self, path: str):
        torch.save({'schema_version': 1, 'data': self.data,
                    'reward_groups': self.reward_groups,
                    'entropy_scores': self.entropy_scores,
                    'rng_state': self.generator.get_state()}, path)

    def load(self, path: str):
        state = torch.load(path, weights_only=False)
        if state.get('schema_version') != 1:
            raise ValueError("This checkpoint lacks full reward-group history. "
                             "Start a fresh run with trainer.resume_mode=disable.")
        self.data = state['data']
        self.reward_groups = state['reward_groups']
        self.entropy_scores = state['entropy_scores']
        self.generator.set_state(state['rng_state'])

    def __len__(self):
        return len(self.data)
