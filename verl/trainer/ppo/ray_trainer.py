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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import json
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
import time

import shutil
from os.path import join as osj

import ray
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics, compute_reward_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.tracking import ValidationGenerationsLogger
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.trainer.ttis.ttis import filter_zero_advantage, select_batch_slice

from verl.trainer.ttis.presampling_selector import DataProfiler
from verl.trainer.ttis.hive_selection import (
    normalize_entropies, prompt_entropy_scores, select_prompt_indices,
    next_candidate_target, update_exploration,
)

WorkerType = Type[Worker]

class Role(Enum):
    ActorRollout = 2

class AdvantageEstimator(str, Enum):
    GRPO = 'grpo'

@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get('GPU', 0) for node, node_info in node_available_resources.items()}

        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes} cannot be satisfied in this ray cluster"
                )

import torch

def compute_advantage(data: DataProto, adv_estimator):
    """Compute the paper's group-relative outcome advantages."""
    if adv_estimator != AdvantageEstimator.GRPO:
        raise ValueError("Only GRPO is supported by the main-study trainer")
    response_length = data.batch['responses'].size(-1)
    advantages, returns, reward = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=data.batch['token_level_rewards'],
        eos_mask=data.batch['attention_mask'][:, -response_length:],
        index=data.non_tensor_batch['uid'],
    )
    data.batch['advantages'] = advantages
    data.batch['returns'] = returns
    data.batch['reward'] = reward
    return data

@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timing_raw.get(name, 0.0) + timer.last

class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'role_worker_mapping.keys()={role_worker_mapping.keys()}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()
        self.data_profiler = DataProfiler(
            fusion_weight=self.config.data.get('history_fusion_weight', 0.9),
            reward_floor=self.config.data.get('history_reward_floor', 0.01),
            seed=self.config.data.get('seed', 42),
        )
        self.start_epoch = 0
        self.current_epoch = 0

        self.targeted_easy = self.config.data.get('target_zero_variance', 0.25) / 3
        self.targeted_hard = self.config.data.get('target_zero_variance', 0.25) * 2 / 3

        if self.config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
            raise ValueError("The HIVE main-study release uses GRPO.")
        if self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
            raise ValueError("The paper's main configuration does not use a KL penalty.")

        self._validate_config()
        self._create_dataloader()

        self.p_easy = self.config.data.p_easy
        self.p_hard = self.config.data.p_hard

        self.prompt_entropies = {}
        self.prompt_normalized_entropies = {}

        # Response entropy is normalized over the current metadata pool.

    def _validate_config(self):
        config = self.config
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")


            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor policy, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, \
                "validation gen temperature should be greater than 0 when enabling do_sample"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         processor=self.processor,
                                         prompt_key=self.config.data.prompt_key,
                                         image_key=self.config.data.get('image_key', 'images'),
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation=self.config.data.get('truncation', 'error'),
                                         filter_overlong_prompts=self.config.data.filter_overlong_prompts)
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            seed = int(self.config.data.get('seed', 42))
            train_dataloader_generator.manual_seed(seed)

            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_size=self.config.data.train_batch_size,
                                                   num_workers=8,
                                                   drop_last=True,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       processor=self.processor,
                                       prompt_key=self.config.data.prompt_key,
                                       image_key=self.config.data.get('image_key', 'images'),
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error',
                                       filter_overlong_prompts=self.config.data.filter_overlong_prompts)
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(
            self.val_dataloader
        ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        import numpy as np

        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])

        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[:generations_to_log]

        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []

        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        print('Repeated validation with n={}'.format(self.config.actor_rollout_ref.rollout.val_kwargs.n))
        print('Validation config:')
        pprint(self.config.actor_rollout_ref.rollout.val_kwargs)

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                                           interleave=True)

            input_ids = test_batch.batch['input_ids']
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )

            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            }
            print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            reward_tensor = self.val_reward_fn(test_batch)

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()
        data_sources = np.concatenate(data_source_lst, axis=0)

        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        sum_acc = 0
        sum_count = 0
        for data_source, rewards in data_source_reward.items():
            rewards = np.array(rewards)
            rewards[rewards < 0.95] = 0
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)
            print(f'>>> Test: val/test_score/{data_source}: {np.mean(rewards)}')
            sum_acc += np.mean(rewards)
            sum_count += 1
        metric_dict['val/test_score/average'] = sum_acc / sum_count
        print(f'>>> Test: val/test_score/average: {sum_acc / sum_count}')

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            self.wg_dicts.append(wg_dict)

        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        dataprofiler_local_path = os.path.join(local_global_step_folder, 'data_profiler.pt')
        if self.data_profiler is not None:
            self.data_profiler.save(dataprofiler_local_path)

        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

        local_current_epoch = os.path.join(self.config.trainer.default_local_dir, 'current_epoch.txt')
        with open(local_current_epoch, 'w') as f:
            f.write(str(self.current_epoch))

        filtering_dynamics = {
            'strategy': 'all_probabilistic',
            'p_easy': self.p_easy,
            'p_hard': self.p_hard
        }
        filtering_dynamics_local_path = os.path.join(local_global_step_folder, 'filtering_dynamics.pt')
        torch.save(filtering_dynamics, filtering_dynamics_local_path)

        local_sampling_num = os.path.join(self.config.trainer.default_local_dir, 'sampling_num.txt')
        with open(local_sampling_num, 'w') as f:
            f.write(str(self.sampling_num))

        normalized_entropies_local_path = os.path.join(local_global_step_folder, 'prompt_normalized_entropies.pt')
        torch.save(self.prompt_normalized_entropies, normalized_entropies_local_path)
        torch.save(self.prompt_entropies, os.path.join(local_global_step_folder, 'response_entropies.pt'))
        torch.save(torch.get_rng_state(), os.path.join(local_global_step_folder, 'selector_rng.pt'))
        with open(os.path.join(local_global_step_folder, 'run_state.json'), 'w') as stream:
            json.dump({'epoch': self.current_epoch, 'sampling_num': self.sampling_num}, stream)

        n=2
        if self.config.trainer.get('keep_all_ckpts', False):
            return

        checkpoint_dirs = [d for d in os.listdir(self.config.trainer.default_local_dir)
                        if d.startswith('global_step_')]
        steps = []
        for d in checkpoint_dirs:
            try:
                step = int(d.split('_')[-1])
                steps.append(step)
            except ValueError:
                continue

        steps.sort()

        if len(steps) > n:
            for step in steps[:-n]:
                dir_to_remove = osj(self.config.trainer.default_local_dir, f'global_step_{step}')
                print(f"Removing old checkpoint directory: {dir_to_remove}")
                shutil.rmtree(dir_to_remove, ignore_errors=True)

    def _load_checkpoint(self):
        print('self.config.trainer.resume_mode', self.config.trainer.resume_mode)
        if self.config.trainer.resume_mode == 'disable':
            return 0

        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        self.data_profiler.load(os.path.join(global_step_folder, 'data_profiler.pt'))

        local_current_epoch = os.path.join(self.config.trainer.default_local_dir, 'current_epoch.txt')
        if os.path.exists(local_current_epoch):
            with open(local_current_epoch, 'r') as f:
                self.start_epoch = int(f.read())
            print(f'Loaded current epoch: {self.start_epoch}')
        else:
            print(f'Warning: No current epoch found at {local_current_epoch}, set epoch = 0')
            self.start_epoch = 0

        filtering_dynamics_local_path = os.path.join(global_step_folder, 'filtering_dynamics.pt')
        if os.path.exists(filtering_dynamics_local_path):
            filtering_dynamics = torch.load(filtering_dynamics_local_path)
            self.p_easy = filtering_dynamics['p_easy']
            self.p_hard = filtering_dynamics['p_hard']
            print(f'Loaded filtering dynamics: {filtering_dynamics}')
        else:
            print(f'Warning: No filtering dynamics found at {filtering_dynamics_local_path}, use default from config.')
            print(f'self.p_easy: {self.p_easy}, self.p_hard: {self.p_hard}')

        local_sampling_num = os.path.join(self.config.trainer.default_local_dir, 'sampling_num.txt')
        if os.path.exists(local_sampling_num):
            with open(local_sampling_num, 'r') as f:
                self.sampling_num = int(f.read())
            print(f'Loaded sampling num: {self.sampling_num}')
        else:
            print(f'Warning: No sampling num found at {local_sampling_num}, set sampling num = 0')
            self.sampling_num = 0

        normalized_entropies_path = os.path.join(global_step_folder, 'prompt_normalized_entropies.pt')
        if os.path.exists(normalized_entropies_path):
            self.prompt_normalized_entropies = torch.load(normalized_entropies_path)
            print(f'Loaded {len(self.prompt_normalized_entropies)} normalized prompt entropies from checkpoint')
        else:
            print(f'Warning: No normalized prompt entropies found at {normalized_entropies_path}, starting fresh')
            self.prompt_normalized_entropies = {}

        rng_path = os.path.join(global_step_folder, 'selector_rng.pt')
        if os.path.exists(rng_path):
            torch.set_rng_state(torch.load(rng_path))
        raw_entropy_path = os.path.join(global_step_folder, 'response_entropies.pt')
        if os.path.exists(raw_entropy_path):
            self.prompt_entropies = torch.load(raw_entropy_path, weights_only=False)

        with open(os.path.join(global_step_folder, 'run_state.json')) as stream:
            state = json.load(stream)
        self.start_epoch = state['epoch']
        self.sampling_num = state['sampling_num']

        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _write_metrics(self, metrics):
        path = self.config.trainer.get('metrics_path')
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            record = {'step': self.global_steps, **{key: float(value) for key, value in metrics.items()}}
            with open(path, 'a') as stream:
                stream.write(json.dumps(record) + '\n')

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0
        self.sampling_num = 0

        self._load_checkpoint()
        if not self.config.trainer.get('val_only', False) and self.global_steps >= self.total_training_steps:
            print(f'Training already reached {self.global_steps} updates.')
            return
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', False) and self.global_steps == 0:
            print('Start Initial Eval...')
            start = time.time()
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            self._write_metrics(val_metrics)
            if self.config.trainer.get('val_only', False):
                return
            print(f'Initial validation time: {time.time() - start}')

        self.global_steps += 1
        last_val_metrics = None

        if hasattr(self.config.data, 'real_train_batch_size'):
            real_train_batch_size = self.config.data.real_train_batch_size * self.config.actor_rollout_ref.rollout.n
        else:
            real_train_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        dynamic_sampling_batch = None

        adaptive_sampling_bs = self.config.data.get('sampling_batch_size', 0)
        default_sampling_batch_size = self.config.data.get('sampling_batch_size', 0)
        pending_history = []

        easy_data_num = 0
        hard_data_num = 0
        total_num = 0

        if self.config.trainer.get('max_steps', -1) > 0:
            self.total_training_steps = self.config.trainer.max_steps
            print(f"Set total training steps to {self.total_training_steps}.")

        refresh_generation_time_flag = True
        generation_time = 0

        acc_batch = None
        easy_data_ratio = 0
        metrics, timing_raw = {}, {}
        step_start = time.perf_counter()
        for epoch in range(self.start_epoch, self.config.trainer.total_epochs):
            self.current_epoch = epoch

            for batch_dict in self.train_dataloader:
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                if refresh_generation_time_flag:
                    generation_time = 0
                    refresh_generation_time_flag = False
                    generation_start = time.time()

                self.data_profiler.set_entropy_scores(self.prompt_normalized_entropies)
                batch = self.data_profiler.filter_examples_all_probabilistic(
                    epoch, batch, self.p_easy, self.p_hard)

                if len(batch) == 0:
                    continue
                non_tensor_keys = [key for key in ('raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs')
                                   if key in batch.non_tensor_batch]
                prompt_batch = batch.select(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=non_tensor_keys,
                )
                padded, pad_size = pad_dataproto_to_divisor(prompt_batch, self.actor_rollout_wg.world_size)
                with _timer('prompt_verification', timing_raw):
                    scored = self.actor_rollout_wg.compute_prompt_entropy(padded)
                if pad_size:
                    scored = unpad_dataproto(scored, pad_size)
                scores = prompt_entropy_scores(
                    scored.batch['entropies'].float().cpu().numpy(),
                    prompt_batch.batch['attention_mask'].cpu().numpy(),
                )
                selected = select_prompt_indices(
                    scores,
                    keep_ratio=self.config.data.get('entropy_keep_ratio', 0.5),
                    offset_ratio=self.config.data.get('entropy_start_ratio', 0.0),
                    group_size=self.config.actor_rollout_ref.rollout.n,
                )
                if not selected:
                    continue
                batch = batch.select_via_index(selected)
                metrics['HIVE/prompt_entropy_threshold'] = float(min(scores[selected]))
                metrics['HIVE/keep_count'] = len(selected)

                if adaptive_sampling_bs > 0:
                    acc_batch = batch if acc_batch is None else DataProto.concat([acc_batch, batch])
                    if len(acc_batch) < adaptive_sampling_bs:
                        continue
                    batch, acc_batch = acc_batch, None
                remainder = len(batch) % self.config.trainer.n_gpus_per_node
                batch = select_batch_slice(batch, len(batch) - remainder)
                print(f'Dynamic Filtering for Sampling: len(batch): {len(batch)}')
                metrics['HIVE/easy_data_ratio'] = easy_data_ratio
                metrics['HIVE/1_n_rollout_num'] = len(batch)

                if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
                    non_tensor_keys = ['multi_modal_data', 'multi_modal_inputs']
                    if 'raw_prompt_ids' in batch.non_tensor_batch.keys():
                        non_tensor_keys.append('raw_prompt_ids')
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=non_tensor_keys,
                    )
                else:
                    non_tensor_keys = []
                    if 'raw_prompt_ids' in batch.non_tensor_batch.keys():
                        non_tensor_keys.append('raw_prompt_ids')
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=non_tensor_keys,
                    )
                is_last_step = self.global_steps >= self.total_training_steps

                with _timer('step', timing_raw):
                    print('start generation...')
                    start = time.time()
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    print(f'generation time: {time.time() - start}')

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)

                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    self.sampling_num += len(batch)

                    with _timer('adv', timing_raw):

                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor

                        batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator)

                    reward_metrics = compute_reward_metrics(batch, self.config)
                    metrics.update(reward_metrics)
                    batch_num = len(batch)

                    easy_data_num += reward_metrics["examples/all_correct_example_ratio"] * batch_num
                    hard_data_num += reward_metrics["examples/format_example_ratio"] * batch_num
                    total_num += batch_num

                    metrics.update(compute_data_metrics(batch=batch))

                    log_index_tensor = torch.tensor(batch.non_tensor_batch['index'].astype(int)).view(-1, self.config.actor_rollout_ref.rollout.n).max(dim=1)[0]
                    log_data_source = batch.non_tensor_batch['data_source'].tolist()[::self.config.actor_rollout_ref.rollout.n]
                    log_reward_tensor = batch.batch['reward'].view(-1, self.config.actor_rollout_ref.rollout.n)

                    data_id_list = self.data_profiler.get_data_id_list(log_data_source, log_index_tensor.tolist())
                    response_means = [None] * len(data_id_list)
                    with _timer('old_log_prob', timing_raw):
                        batch = batch.union(self.actor_rollout_wg.compute_log_prob(batch))
                    length = batch.batch['responses'].shape[1]
                    mask = batch.batch['attention_mask'][:, -length:]
                    means = (batch.batch['entropies'] * mask).sum(-1) / mask.sum(-1).clamp_min(1)
                    response_means = means.view(-1, self.config.actor_rollout_ref.rollout.n).mean(-1).tolist()
                    for data_id, rewards, entropy in zip(data_id_list, log_reward_tensor, response_means):
                        pending_history.append((epoch, data_id, rewards.cpu().numpy().copy(), entropy))

                    batch = filter_zero_advantage(batch, self.config)
                    if dynamic_sampling_batch is None:
                        dynamic_sampling_batch = batch
                    else:
                        dynamic_sampling_batch = DataProto.concat([dynamic_sampling_batch, batch])

                    if len(dynamic_sampling_batch) < real_train_batch_size:
                        missing = (real_train_batch_size - len(dynamic_sampling_batch)) // self.config.actor_rollout_ref.rollout.n
                        adaptive_sampling_bs = next_candidate_target(
                            missing, reward_metrics['examples/non_diverse_examples_ratio'],
                            default_sampling_batch_size,
                            minimum=self.config.data.get('resample_minimum', 64),
                            safety_factor=self.config.data.get('resample_safety_factor', 1.25),
                        )
                        continue
                    batch = select_batch_slice(dynamic_sampling_batch, real_train_batch_size)
                    dynamic_sampling_batch = None
                    adaptive_sampling_bs = default_sampling_batch_size

                    generation_time += time.time() - generation_start
                    refresh_generation_time_flag = True
                    print(f'Total generation time: {generation_time}')
                    metrics.update({'timing_s/total_generation_time': generation_time})

                    hard_data_ratio = hard_data_num / total_num
                    easy_data_ratio = easy_data_num / total_num

                    examples_log = {
                        'examples/too_hard_ratio': hard_data_ratio,
                        'examples/too_easy_ratio': easy_data_ratio,
                        'examples/total_sampled': total_num,
                    }

                    examples_log['examples/p_easy'] = self.p_easy
                    examples_log['examples/p_hard'] = self.p_hard
                    print(f'p_easy: {self.p_easy}, p_hard: {self.p_hard}')

                    metrics.update(examples_log)

                    easy_data_num = 0
                    hard_data_num = 0
                    total_num = 0

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()
                    if 'old_log_probs' not in batch.batch:
                        with _timer('old_log_prob', timing_raw):
                            batch = batch.union(self.actor_rollout_wg.compute_log_prob(batch))
                    print('start update actor...')
                    start = time.time()
                    with _timer('update_actor', timing_raw):
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                    metrics.update(actor_output_metrics)

                    print(f'update actor time: {time.time() - start}')

                    # Commit metadata only after the policy update (Algorithm 1).
                    for visit_epoch, data_id, rewards, entropy in pending_history:
                        self.data_profiler.add_reward_group(visit_epoch, data_id, rewards)
                        if entropy is not None:
                            self.prompt_entropies[data_id] = entropy
                    pending_history.clear()
                    self.prompt_normalized_entropies = normalize_entropies(self.prompt_entropies)
                    self.data_profiler.set_entropy_scores(self.prompt_normalized_entropies)
                    options = dict(step=self.config.data.get('exploration_step', 0.01),
                                   minimum=self.config.data.get('min_p', 0.05),
                                   maximum=self.config.data.get('max_p', 0.95))
                    self.p_easy = update_exploration(self.p_easy, easy_data_ratio, self.targeted_easy, **options)
                    self.p_hard = update_exploration(self.p_hard, hard_data_ratio, self.targeted_hard, **options)
                    metrics.update({'examples/p_easy': self.p_easy, 'examples/p_hard': self.p_hard})

                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        (is_last_step or  self.global_steps % self.config.trainer.test_freq == 0):
                        print('start validation...')
                        start = time.time()
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)
                        print(f'validation time: {time.time() - start}')

                    if self.config.trainer.save_freq > 0 and ( is_last_step or \
                            self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                timing_raw['step'] = time.perf_counter() - step_start
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                sampling_metircs = {}
                sampling_metircs['sampling/sampling_total_num'] = self.sampling_num
                metrics.update(sampling_metircs)

                logger.log(data=metrics, step=self.global_steps)
                self._write_metrics(metrics)
                metrics, timing_raw = {}, {}
                step_start = time.perf_counter()

                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    return

                print(f'epoch: {epoch}')
                print(f'Global_steps done: {self.global_steps}')
                print(f'sampling_num: {self.sampling_num}')
                self.global_steps += 1
                print(f'Start global_steps: {self.global_steps}')

            print(f'Epoch {epoch} done!')
