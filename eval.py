"""
Evaluates a given policy on some number of episodes
"""
import argparse
from atcenv import Environment
from pathlib import Path
import torch
from tqdm import tqdm

def parse_args(parser: argparse.ArgumentParser):
    parser.add_argument('policy', choices=['baseline'], help='The policy to evaluate')
    parser.add_argument('num_episodes', type=int)
    parser.add_argument('num_planes', type=int, help='The number of planes to evaluate on')
    parser.add_argument('save_path', type=str, help='The path to save results to')
    parser.add_argument('-d', '--debug', action='store_true')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args(argparse.ArgumentParser())
    
    num_planes = args.num_planes
    policy_name = args.policy
    num_episodes = args.num_episodes

    if policy_name == 'baseline':
        policy = lambda state: [[0, 0] for _ in range(num_planes)]

    env = Environment()
    
    num_los = [0 for _ in range(num_episodes)]
    episode_lengths = [0 for _ in range(num_episodes)]

    for episode in tqdm(range(num_episodes)):
        obs = env.reset(args.num_planes)
        
        done = False
        
        # Holds the unique losses of separation in the episode
        episode_los = set()

        while not done:
            action = policy(obs)

            new_obs, rew, term, trunc, _ = env.step(action)

            episode_los = episode_los.union(env.conflicts)

            episode_lengths[episode] += 1
            
            done = term or trunc

            if args.debug:
                env.render()

        num_los[episode] = len(episode_los)

        if args.debug:
            env.close()

    save_path = Path(args.save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({'num_conflicts':num_los, 'episode_lengths': episode_lengths}, save_path)
