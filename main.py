"""
Example
"""

import numpy as np
import random
from atcenv import Environment
from tqdm import tqdm
import argparse
from jsonargparse import ActionConfigFile
from agents.ppo import PPO
import torch

def parse_args(parser: argparse.ArgumentParser):
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--config', action=ActionConfigFile)
    parser.add_argument('--a_lr', type=float, default=.00001)
    parser.add_argument('--c_lr', type=float, default=.00001)
    parser.add_argument('--trajectories_per_batch', type=int, default=12000)
    parser.add_argument('--updates_per_batch', type=int, default=5)
    parser.add_argument('--minibatch_size', type=int, default=256)
    parser.add_argument('--gamma', type=float, default=.99)
    # parse arguments
    return parser.parse_args()

if __name__ == "__main__":
    random.seed(52)
    
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    parser = argparse.ArgumentParser(
        prog='Conflict resolution environment',
        description='Basic conflict resolution environment for training policies with reinforcement learning',
    )
    
    args = parse_args(parser)

    # init environment
    env = Environment(num_flights=10)

    RL = PPO()
    
    metrics = RL.train(env, args.episodes, args.a_lr, args.c_lr, args.trajectories_per_batch, args.updates_per_batch, args.minibatch_size, args.gamma, device)
