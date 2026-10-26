import torch

def select_batch_slice(batch, num):
    selected_indices = torch.arange(num)
    selected_batch = batch.select_via_index(selected_indices)

    return selected_batch


def filter_zero_advantage(batch, config):
    reward = batch.batch['reward']
    reward = reward.reshape(-1, config.actor_rollout_ref.rollout.n)
    zero_advantage_flag = (reward.max(dim=1)[0] == reward.min(dim=1)[0])
    selected_indices = torch.arange(reward.shape[0])[~zero_advantage_flag]
    print('****************** Q-Scaling ******************')
    print('TTIS: Filter Zero Advantage')
    print(f'Original batch size: {len(batch)}')
    print('zero_advantage_flag:')
    print(zero_advantage_flag)
    selected_indices = selected_indices.unsqueeze(1) * config.actor_rollout_ref.rollout.n + torch.arange(config.actor_rollout_ref.rollout.n).unsqueeze(0)
    batch = batch.select_via_index(selected_indices.view(-1))
    print(f'New batch size: {len(batch)}')
    return batch
