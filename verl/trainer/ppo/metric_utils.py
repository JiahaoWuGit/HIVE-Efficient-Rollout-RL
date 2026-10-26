# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Metrics related to the PPO trainer.
"""

import torch
from typing import Any, Dict, List
import numpy as np
from verl import DataProto


def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto) -> Dict[str, Any]:
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)


    metrics = {
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),

        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['adv', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    total_num_tokens = sum(batch.meta_info['global_token_num'])
    time = timing_raw['step']
    return {
        'perf/total_num_tokens': total_num_tokens,
        'perf/time_per_step': time,
        'perf/throughput': total_num_tokens / (time * n_gpus),
    }


def compute_reward_metrics(batch, config):
    reward_tensor = batch.batch['token_level_scores'].sum(-1)

    reward_metrics = {}

    if True:
        reward_metrics["reward/mean"] = torch.mean(reward_tensor).detach().item()
        all_correct = torch.sum(reward_tensor == 1).float() / reward_tensor.numel()
        reward_metrics["reward/all_correct_ratio"] = all_correct.detach().item()
        format_error = torch.sum(reward_tensor == 0).float() / reward_tensor.numel()
        reward_metrics["reward/format_error_ratio"] = format_error.detach().item()
        format_error = torch.sum(reward_tensor == 0.1).float() / reward_tensor.numel()
        reward_metrics["reward/wrong_answer_ratio"] = format_error.detach().item()

        prompt_num = reward_tensor.view(-1, config.actor_rollout_ref.rollout.n).size(0)

        print('reward:')
        print(reward_tensor.view(-1, config.actor_rollout_ref.rollout.n).shape)
        print(reward_tensor.view(-1, config.actor_rollout_ref.rollout.n))
        print('Mean reward:', torch.mean(reward_tensor).detach().item())

        non_diverse_examples = (reward_tensor.view(-1, config.actor_rollout_ref.rollout.n).max(-1)[0] == reward_tensor.view(-1, config.actor_rollout_ref.rollout.n).min(-1)[0]).sum()

        non_diverse_examples_ratio = non_diverse_examples / prompt_num
        print('non_diverse_examples:', non_diverse_examples)
        reward_metrics["examples/non_diverse_examples_ratio"] = non_diverse_examples_ratio.detach().item()

        format_example = torch.logical_and(reward_tensor.view(-1, config.actor_rollout_ref.rollout.n).max(-1)[0] == 0.1, reward_tensor.view(-1, config.actor_rollout_ref.rollout.n).min(-1)[0] == 0.1).sum()
        format_example = format_example / prompt_num

        all_correct_example = torch.logical_and(reward_tensor.view(-1, config.actor_rollout_ref.rollout.n).max(-1)[0] == 1, reward_tensor.view(-1, config.actor_rollout_ref.rollout.n).min(-1)[0] == 1).sum()
        all_correct_example = all_correct_example / prompt_num

        reward_metrics["examples/format_example_ratio"] = format_example.detach().item()
        reward_metrics["examples/all_correct_example_ratio"] = all_correct_example.detach().item()

        easy_examples_ratio = (reward_tensor.view(-1, config.actor_rollout_ref.rollout.n).mean(-1) > 0.75).sum() / prompt_num
        reward_metrics["examples/easy_examples_ratio_0.75"] = easy_examples_ratio.detach().item()

    return reward_metrics
