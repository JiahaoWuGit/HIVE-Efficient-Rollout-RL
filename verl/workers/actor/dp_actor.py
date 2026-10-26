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
Single Process Actor
"""


import itertools
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get('use_torch_compile', True)
            else verl_F.entropy_from_logits)

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        multi_modal_inputs = {}
        if 'multi_modal_inputs' in micro_batch:
            for key in micro_batch['multi_modal_inputs'][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch['multi_modal_inputs']],
                                                    dim=0)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."),
                                                          indices).transpose(0, 1).unsqueeze(
                                                              1)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                          indices).transpose(0, 1)

                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           **multi_modal_inputs,
                                           use_cache=False)
                logits_rmpad = output.logits.squeeze(0)

                logits_rmpad.div_(temperature)

                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)

                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                if self.use_ulysses_sp:
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]

            else:
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           **multi_modal_inputs,
                                           use_cache=False)
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1, :]
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm


    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
            entropy_lst.append(entropy)
        log_probs = torch.concat(log_probs_lst, dim=0)
        entropy = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            entropy = entropy[revert_indices]

        return log_probs, entropy

    def update_policy(self, data: DataProto):
        self.actor_module.train()

        temperature = data.meta_info['temperature']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())
                    responses = data['responses']
                    response_length = responses.size(1)
                    attention_mask = data['attention_mask']
                    response_mask = attention_mask[:, -response_length:]
                    old_log_prob = data['old_log_probs']
                    advantages = data['advantages']

                    clip_ratio = self.config.clip_ratio
                    entropy_coeff = self.config.entropy_coeff

                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                                  log_prob=log_prob,
                                                                                  advantages=advantages,
                                                                                  eos_mask=response_mask,
                                                                                  cliprange=clip_ratio)
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)

                    policy_loss = pg_loss - entropy_loss * entropy_coeff


                    if self.config.use_dynamic_bsz:
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        'actor/entropy_loss': entropy_loss.detach().item(),
                        'actor/pg_loss': pg_loss.detach().item(),
                        'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                        'actor/ppo_kl': ppo_kl.detach().item(),
                    }
                    append_to_dict(metrics, data)

                print('>>> Weight update!!!')
                grad_norm = self._optimizer_step()
                data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics

    def compute_prompt_entropy(self, prompt_data: DataProto) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the per-position entropy and log-prob of the prompt.

        Args:
            prompt_data: DataProto containing:
                - input_ids: [batch_size, prompt_length] - prompt token ids
                - attention_mask: [batch_size, prompt_length]
                - position_ids: [batch_size, prompt_length]
                - meta_info with:
                    - micro_batch_size: micro batch size (optional, default 8)
                    - temperature: temperature (optional, default 1.0)
                    - use_dynamic_bsz: whether to use dynamic batch size (optional, default False)
                    - max_token_len: max token length (when use_dynamic_bsz=True)

        Returns:
            entropies: [batch_size, prompt_length] - per-position prompt entropy
            log_probs: [batch_size, prompt_length] - per-position prompt log-prob
                    - log_probs[:, 0] = 0 (the first position has no preceding token)
                    - log_probs[:, i] = log P(token[i] | prompt[:i]) for i > 0
        """
        self.actor_module.eval()

        micro_batch_size = prompt_data.meta_info.get('micro_batch_size', 8)
        temperature = prompt_data.meta_info.get('temperature', 1.0)
        use_dynamic_bsz = prompt_data.meta_info.get('use_dynamic_bsz', False)

        prompt_input_ids = prompt_data.batch['input_ids']
        prompt_attention_mask = prompt_data.batch['attention_mask']
        prompt_position_ids = prompt_data.batch['position_ids']

        if prompt_input_ids.shape[0] == 0:
            prompt_length = prompt_input_ids.shape[1] if len(prompt_input_ids.shape) > 1 else 0
            device = prompt_input_ids.device
            empty_entropies = torch.empty((0, prompt_length), device=device, dtype=torch.float32)
            empty_log_probs = torch.empty((0, prompt_length), device=device, dtype=torch.float32)
            return empty_entropies, empty_log_probs

        has_multi_modal_inputs = 'multi_modal_inputs' in prompt_data.non_tensor_batch.keys()

        select_keys = ['input_ids', 'attention_mask', 'position_ids']
        batch = prompt_data.select(batch_keys=select_keys).batch

        if has_multi_modal_inputs:
            batch_size = prompt_data.batch.batch_size[0]
            num_micro_batches = prompt_data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            if num_micro_batches == 0:
                num_micro_batches = 1
            if batch_size % num_micro_batches != 0:
                micro_batches = batch.split(micro_batch_size)
            else:
                non_tensor_select_keys = ['multi_modal_inputs']
                micro_batches = prompt_data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)

        elif use_dynamic_bsz:
            max_token_len = prompt_data.meta_info.get('max_token_len', 2048) * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        entropy_lst = []
        log_probs_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            if 'input_ids' in micro_batch:
                if micro_batch['input_ids'].shape[0] == 0:
                    continue

            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch_for_prompt_entropy_and_logprob(
                    micro_batch, temperature=temperature
                )
            entropy_lst.append(entropy)
            log_probs_lst.append(log_probs)


        if len(entropy_lst) == 0:
            prompt_length = prompt_input_ids.shape[1]
            device = prompt_input_ids.device
            empty_entropies = torch.empty((0, prompt_length), device=device, dtype=torch.float32)
            empty_log_probs = torch.empty((0, prompt_length), device=device, dtype=torch.float32)
            return empty_entropies, empty_log_probs

        entropies = torch.concat(entropy_lst, dim=0)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == entropies.size(0), f"{len(indices)} vs. {entropies.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            entropies = entropies[revert_indices]
            log_probs = log_probs[revert_indices]

        return entropies, log_probs


    def _forward_micro_batch_for_prompt_entropy_and_logprob(
        self, micro_batch, temperature
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-position prompt entropy and log-probs for a single micro batch.

        Args:
            micro_batch: dict containing input_ids, attention_mask, position_ids
            temperature: temperature parameter

        Returns:
            entropies: [batch_size, prompt_length] - per-position entropy
            log_probs: [batch_size, prompt_length] - per-position log-prob
        """
        multi_modal_inputs = {}
        if 'multi_modal_inputs' in micro_batch:
            for key in micro_batch['multi_modal_inputs'][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch['multi_modal_inputs']], dim=0)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if batch_size == 0:
                device = input_ids.device
                dtype = torch.float32
                empty_entropy = torch.zeros((0, seqlen), dtype=dtype, device=device)
                empty_log_probs = torch.zeros((0, seqlen), dtype=dtype, device=device)
                return empty_entropy, empty_log_probs

            valid_tokens = attention_mask.sum()
            if valid_tokens.item() == 0:
                device = input_ids.device
                dtype = torch.float32
                empty_entropy = torch.zeros((batch_size, seqlen), dtype=dtype, device=device)
                empty_log_probs = torch.zeros((batch_size, seqlen), dtype=dtype, device=device)
                return empty_entropy, empty_log_probs

            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids, "c b s ... -> (b s) c ..."),
                        indices
                    ).transpose(0, 1).unsqueeze(1)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                        indices
                    ).transpose(0, 1)

                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size
                    )
                else:
                    pass

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False
                )
                logits_rmpad = output.logits.squeeze(0)

                logits_rmpad.div_(temperature)

                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)

                log_probs_rmpad = logprobs_from_logits(
                    logits=logits_rmpad,
                    labels=input_ids_rmpad_rolled.squeeze(0) if self.use_ulysses_sp else input_ids_rmpad_rolled.squeeze(0)
                )

                if self.use_ulysses_sp:
                    entropy_rmpad = gather_outpus_and_unpad(
                        entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )
                    log_probs_rmpad = gather_outpus_and_unpad(
                        log_probs_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )

                full_entropy = pad_input(
                    hidden_states=entropy_rmpad.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen
                )
                full_log_probs = pad_input(
                    hidden_states=log_probs_rmpad.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen
                )

                entropy = full_entropy.squeeze(-1)
                log_probs = full_log_probs.squeeze(-1)

                log_probs[:, 0] = 0.0

            else:
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False
                )
                logits = output.logits
                logits.div_(temperature)

                entropy = verl_F.entropy_from_logits(logits)

                if seqlen > 1:
                    logits_for_logprob = logits[:, :-1, :]
                    labels = input_ids[:, 1:]
                    log_probs_partial = logprobs_from_logits(logits_for_logprob, labels)

                    first_pos_log_probs = torch.zeros(
                        batch_size, 1,
                        device=log_probs_partial.device,
                        dtype=log_probs_partial.dtype
                    )
                    log_probs = torch.cat([first_pos_log_probs, log_probs_partial], dim=1)
                else:
                    log_probs = torch.zeros(
                        batch_size, 1,
                        device=entropy.device,
                        dtype=entropy.dtype
                    )

            return entropy, log_probs
