import torch
from collections import deque
import numpy as np

class TrajectoryBuffer(torch.utils.data.Dataset):
    '''
    Creates a trajectory buffer that can be used to train certain
    reinforcement learning problems
    '''
    def __init__(self, maxlen: int):
        self.buffer = deque([], maxlen=maxlen)
        self.maxlen = maxlen

    def add_item(self, 
                       obs: np.ndarray, 
                       action: np.ndarray, 
                       reward: np.ndarray,
                       reward_to_go: np.ndarray,
                       next_obs: np.ndarray,
                       log_prob: np.ndarray,
                       done: np.ndarray):
        
        # Convert everything to Tensors now rather than later
        obs_t = torch.Tensor(obs)
        action_t = torch.Tensor(action)
        reward_t = torch.Tensor([reward])
        reward_to_go_t = torch.Tensor([reward_to_go])
        next_obs_t = torch.Tensor(next_obs)
        log_prob_t = torch.Tensor(log_prob)
        done_t = torch.Tensor(done)

        self.buffer.append((obs_t, action_t, reward_t, reward_to_go_t, next_obs_t, log_prob_t, done_t))

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        obs, action, reward, reward_to_go, next_obs, log_prob, done = self.buffer[idx]

        return obs, action, reward, reward_to_go, next_obs, log_prob, done

    def clear(self):
        self.buffer.clear()
