import torch
from collections import deque
import numpy as np

class ReplayBuffer(torch.utils.data.Dataset):
    '''
    Creates a replay buffer that can be used to train certain
    reinforcement learning problems
    '''
    def __init__(self, maxlen: int):
        self.buffer = deque([], maxlen=maxlen)

    def add_experience(self, 
                       obs: np.ndarray, 
                       action: np.ndarray, 
                       reward: np.ndarray, 
                       next_obs: np.ndarray, 
                       done: np.ndarray):
        
        # Convert everything to Tensors now rather than later
        obs_t = torch.Tensor(obs)
        action_t = torch.Tensor(action)
        reward_t = torch.Tensor(reward)
        next_obs_t = torch.Tensor(next_obs)
        done_t = torch.Tensor(done)

        self.buffer.append((obs_t, action_t, reward_t, next_obs_t, done_t))

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        obs, action, reward, next_obs, done = self.buffer[idx]

        return obs, action, reward, next_obs, done
