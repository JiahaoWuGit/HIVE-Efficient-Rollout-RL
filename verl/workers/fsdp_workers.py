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
FSDP actor and rollout workers for HIVE's GRPO training.
"""


import logging
import os
import warnings
import psutil

import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh
from omegaconf import DictConfig, open_dict
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager
from verl.utils.fsdp_utils import offload_fsdp_optimizer, offload_fsdp_model_to_cpu, load_fsdp_optimizer, \
    load_fsdp_model_to_gpu
from verl.utils.import_utils import import_external_libs
from verl.utils.flops_counter import FlopsCounter
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

from codetiming import Timer


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])
    else:
        device_mesh = init_device_mesh('cuda',
                                       mesh_shape=(world_size // fsdp_size, fsdp_size),
                                       mesh_dim_names=['ddp', 'fsdp'])
    return device_mesh


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy
    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


class ActorRolloutRefWorker(Worker):
    """
    Actor and rollout worker with a shared FSDP/vLLM hybrid engine.
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        world_size = torch.distributed.get_world_size()
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self.role = role
        assert self.role in ['actor', 'rollout', 'actor_rollout']

        self._is_actor = self.role in ['actor', 'actor_rollout']
        self._is_rollout = self.role in ['rollout', 'actor_rollout']

        self._is_offload_param = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get('param_offload', False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get('optimizer_offload', False)

        if self._is_actor:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= (self.device_mesh.size() // self.ulysses_sequence_parallel_size)
            assert self.config.actor.ppo_mini_batch_size > 0, f'ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than 0 after normalization'
            if self.config.actor.ppo_micro_batch_size is not None:
                self.config.actor.ppo_micro_batch_size //= (self.device_mesh.size() //
                                                            self.ulysses_sequence_parallel_size)
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size
                assert self.config.actor.ppo_mini_batch_size % self.config.actor.ppo_micro_batch_size_per_gpu == 0, \
                    f'normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be divisible by ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}'
                assert self.config.actor.ppo_mini_batch_size // self.config.actor.ppo_micro_batch_size_per_gpu > 0, \
                    f'normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}'

        if self._is_rollout and self.config.rollout.log_prob_micro_batch_size is not None:
            self.config.rollout.log_prob_micro_batch_size //= (self.device_mesh.size() //
                                                               self.ulysses_sequence_parallel_size)
            self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size

    def _build_model_optimizer(self,
                               model_path,
                               fsdp_config,
                               optim_config,
                               override_model_config,
                               use_remove_padding=False,
                               enable_gradient_checkpointing=False,
                               trust_remote_code=False,
                               use_liger=False,
                               role='actor'):
        from verl.utils.model import print_model_size, update_model_config, get_generation_config
        from verl.utils.torch_dtypes import PrecisionType
        from transformers import AutoModelForCausalLM, AutoConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
        from torch import optim

        assert role == 'actor'

        log_gpu_memory_usage('Before init from HF AutoModel', logger=logger)
        local_path = copy_to_local(model_path)

        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        torch_dtype = fsdp_config.get('model_dtype', None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)

        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(actor_model_config.model_type)

        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(actor_model_config, verbose=True)

        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f'Model config after override: {actor_model_config}')

        init_context = get_init_weight_context_manager(use_meta_tensor=not actor_model_config.tie_word_embeddings,
                                                       mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            actor_module_class = AutoModelForCausalLM

            actor_module = actor_module_class.from_pretrained(pretrained_model_name_or_path=local_path,
                                                              torch_dtype=torch_dtype,
                                                              config=actor_model_config,
                                                              attn_implementation='flash_attention_2',
                                                              trust_remote_code=trust_remote_code)
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
                _apply_liger_kernel_to_instance(model=actor_module)

            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage('After init from HF AutoModel', logger=logger)

        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get('wrap_policy', None))


        print(f'wrap_policy: {auto_wrap_policy}')

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        cpu_offload = None
        actor_module_fsdp = FSDP(
            actor_module,
            cpu_offload=cpu_offload,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=self.device_mesh,
            forward_prefetch=False)

        log_gpu_memory_usage('After Actor FSDP init', logger=logger)

        if role == 'actor' and optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup
            actor_optimizer = optim.AdamW(actor_module_fsdp.parameters(),
                                        lr=optim_config.lr,
                                        betas=optim_config.get('betas', (0.9, 0.999)),
                                        weight_decay=optim_config.get('weight_decay', 1e-2))

            total_steps = optim_config.get('total_training_steps', 0)
            num_warmup_steps = int(optim_config.get('lr_warmup_steps', -1))
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get('lr_warmup_steps_ratio', 0.)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

            actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer,
                                                                   num_warmup_steps=num_warmup_steps)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        log_gpu_memory_usage('After actor optimizer init', logger=logger)

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self):
        from torch.distributed.device_mesh import init_device_mesh
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f'rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}'
        rollout_device_mesh = init_device_mesh('cuda', mesh_shape=(dp, infer_tp), mesh_dim_names=['dp', 'infer_tp'])

        from verl.workers.rollout.vllm_rollout import vLLMRollout
        from verl.workers.sharding_manager import FSDPVLLMShardingManager
        log_gpu_memory_usage('Before building vllm rollout', logger=None)
        local_path = copy_to_local(self.config.model.path)
        rollout = vLLMRollout(model_path=local_path,
                              config=self.config.rollout,
                              tokenizer=self.tokenizer,
                              model_hf_config=self.actor_model_config,
                              device_mesh=rollout_device_mesh)
        log_gpu_memory_usage('After building vllm rollout', logger=None)
        if torch.distributed.get_world_size() == 1:
            self.config.rollout.load_format = 'dummy_hf'
        rollout_sharding_manager = FSDPVLLMShardingManager(module=self.actor_module_fsdp,
                                                           inference_engine=rollout.inference_engine,
                                                           model_config=self.actor_model_config,
                                                           full_params='hf' in self.config.rollout.load_format,
                                                           device_mesh=rollout_device_mesh)
        log_gpu_memory_usage('After building sharding manager', logger=None)

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor
        import_external_libs(self.config.model.get('external_lib', None))

        from omegaconf import OmegaConf
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))

        use_remove_padding = self.config.model.get('use_remove_padding', False)

        if self._is_actor or self._is_rollout:
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                enable_gradient_checkpointing=self.config.model.get('enable_gradient_checkpointing', False),
                trust_remote_code=self.config.model.get('trust_remote_code', False),
                use_liger=self.config.model.get('use_liger', False),
                role='actor')

            self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage('After offload actor optimizer during init', logger=logger)
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
            self.actor = DataParallelPPOActor(config=self.config.actor,
                                              actor_module=self.actor_module_fsdp,
                                              actor_optimizer=self.actor_optimizer)

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout()


        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        data = data.to(torch.cuda.current_device())

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())

        log_gpu_memory_usage('Before update policy', logger=logger)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name='update_policy', logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info['global_token_num']
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics[
                'perf/mfu/actor'] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            metrics['perf/max_memory_allocated_gb'] = torch.cuda.max_memory_allocated() / (1024**3)
            metrics['perf/max_memory_reserved_gb'] = torch.cuda.max_memory_reserved() / (1024**3)
            metrics['perf/cpu_memory_used_gb'] = psutil.virtual_memory().used / (1024**3)

            self.actor_lr_scheduler.step()
            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics['actor/lr'] = lr

            log_gpu_memory_usage('After update policy', logger=logger)

            output = DataProto(meta_info={'metrics': metrics})

            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)

        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        prompts = prompts.to(torch.cuda.current_device())

        assert self._is_rollout
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        meta_info = {
            'eos_token_id':
                self.generation_config.eos_token_id
                if self.generation_config is not None else self.tokenizer.eos_token_id,
            'pad_token_id':
                self.generation_config.pad_token_id
                if self.generation_config is not None else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        with self.rollout_sharding_manager:

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)

            log_gpu_memory_usage('After entering rollout sharding manager', logger=logger)

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)

            log_gpu_memory_usage('After rollout generation', logger=logger)

            output = self.rollout_sharding_manager.postprocess_data(output)

        output = output.to('cpu')

        log_gpu_memory_usage('After recompute log prob', logger=logger)
        return output


    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        data = data.to(torch.cuda.current_device())
        data.meta_info['micro_batch_size'] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info['max_token_len'] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info['temperature'] = self.config.rollout.temperature
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            log_probs, entropies = self.actor.compute_log_prob(data=data)

            output = DataProto.from_dict(tensors={'old_log_probs': log_probs, 'entropies': entropies},
                                         meta_info={'temperature': self.config.rollout.temperature})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to('cpu')
        if self.world_size > 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        log_gpu_memory_usage('After compute_log_prob', logger=logger)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_prompt_entropy(self, prompt_data: DataProto):
        """
        Compute the per-position entropy and log-prob of the prompt.

        Args:
            prompt_data: DataProto containing:
                - input_ids: [batch_size, prompt_length] - prompt token ids
                - attention_mask: [batch_size, prompt_length]
                - position_ids: [batch_size, prompt_length]
                - meta_info with:
                    - micro_batch_size: micro batch size (optional, defaults applied)
                    - temperature: temperature (optional, defaults applied)
                    - use_dynamic_bsz: whether to use dynamic batch size (optional)
                    - max_token_len: max token length (when use_dynamic_bsz=True)

        Returns:
            output: DataProto containing:
                - entropies: [batch_size, prompt_length] - per-position prompt entropy
                - log_probs: [batch_size, prompt_length] - per-position prompt log-prob
        """
        assert self._is_actor, "compute_prompt_entropy requires actor role"

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        prompt_data = prompt_data.to(torch.cuda.current_device())

        if 'micro_batch_size' not in prompt_data.meta_info:
            prompt_data.meta_info['micro_batch_size'] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        if 'max_token_len' not in prompt_data.meta_info:
            prompt_data.meta_info['max_token_len'] = self.config.rollout.log_prob_max_token_len_per_gpu
        if 'use_dynamic_bsz' not in prompt_data.meta_info:
            prompt_data.meta_info['use_dynamic_bsz'] = self.config.rollout.log_prob_use_dynamic_bsz
        if 'temperature' not in prompt_data.meta_info:
            prompt_data.meta_info['temperature'] = self.config.rollout.temperature

        with self.ulysses_sharding_manager:
            prompt_data = self.ulysses_sharding_manager.preprocess_data(prompt_data)
            entropies, log_probs = self.actor.compute_prompt_entropy(prompt_data=prompt_data)

            output = DataProto.from_dict(
                tensors={
                    'entropies': entropies,
                    'log_probs': log_probs
                },
                meta_info={
                    'temperature': prompt_data.meta_info.get('temperature', self.config.rollout.temperature)
                }
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to('cpu')

        if self.world_size > 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        log_gpu_memory_usage('After compute_prompt_entropy', logger=logger)
        return output


    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, remove_previous_ckpt=False):
        assert self._is_actor
        import torch
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.save_checkpoint(local_path=local_path,
                                                hdfs_path=hdfs_path,
                                                global_step=global_step,
                                                remove_previous_ckpt=remove_previous_ckpt)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path, del_local_after_load=False):

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.load_checkpoint(path=path, del_local_after_load=del_local_after_load)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)
