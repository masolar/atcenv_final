import torch
from .trajectory_buffer import TrajectoryBuffer
from .networks import Actor, Critic
import gym
from tqdm import tqdm
import numpy as np
from typing import List

class PPO:
    """
    An implementation of the PPO algorithm for Reinforcement Learning
    """
    def __init__(self):
        # Initialize networks
        self.actor = Actor(15, 1)
        self.critic = Critic(15)
        
    def train(self, 
              env: gym.Env, 
              epochs: int, 
              a_lr: float, 
              c_lr: float, 
              trajectories_per_batch: int, 
              updates_per_batch: int,
              minibatch_size: int,
              gamma: float,
              device: torch.device,
              debug: bool):
        """
        Given an environment, trains the actor and critic on the environment.
        
        Parameters:
            env: The environment to train on
            epochs: The number of epochs to train for
            a_lr: The actor network learning rate
            c_lr: The critic network learning rate
            trajectories_per_batch: The number of trajectories to generate before updating the network
            updates_per_batch: The number of times we use the batch to update the network
            minibatch_size: The size of the minibatches that are used to update the networks each epoch
            gamma: The amount to discount rewards by
            device: The device to place all of our tensors on
        """
        
        self.actor.to(device)
        self.critic.to(device)

        a_opt = torch.optim.Adam(self.actor.parameters(), lr=a_lr)
        c_opt = torch.optim.Adam(self.critic.parameters(), lr=c_lr)
        
        buffer = TrajectoryBuffer(trajectories_per_batch)
        
        # We can use a builtin loss for the critic. The actor needs a custom one
        c_loss_fn = torch.nn.MSELoss()

        # Holds the metrics we'd like to return after training
        metric_rews = []
        metric_a_loss = []
        metric_c_loss = []

        for epoch in tqdm(range(epochs)):
            buffer.clear()

            self._generate_batch(env, buffer, gamma, device, debug)
            
            """
            It's been suggested to train PPO using minibatches to help with memory
            constraints
            """
            for _ in range(updates_per_batch):
                loader = torch.utils.data.DataLoader(buffer, 
                                                     batch_size=minibatch_size,
                                                     shuffle=True,
                                                     generator=torch.Generator(device=device))
                
                avg_rews = []
                avg_a_loss = []
                avg_c_loss = []

                for batch in loader:
                    batch_obs, batch_act, batch_rew, batch_rtg, batch_next_obs, batch_log_prob, batch_done = batch
                    
                    avg_rews += [item[0] for item in batch_rew.tolist()]

                    # Move things to the GPU
                    batch_obs = batch_obs.to(device)
                    batch_act = batch_act.to(device)
                    batch_rtg = batch_rtg.to(device)
                    batch_log_prob = batch_log_prob.to(device)
                    v_curr = self.critic(batch_obs)
                    
                    advantage = batch_rtg - v_curr
                    
                    # Needed for the ratio described in the pseudocode
                    dist = self.actor(batch_obs)
                    log_probs = dist.log_prob(batch_act)

                    ratio = torch.exp(log_probs) / torch.exp(batch_log_prob)
                    
                    # Following the pseudocode
                    a_loss = torch.mean(-torch.min(ratio * advantage, torch.clamp(ratio, .8, 1.2) * advantage))
                    c_loss = c_loss_fn(v_curr, batch_rew)
                    
                    avg_a_loss.append(a_loss.item())
                    avg_c_loss.append(c_loss.item())

                    a_opt.zero_grad()
                    a_loss.backward(retain_graph=True)
                    a_opt.step()

                    c_opt.zero_grad()
                    c_loss.backward()
                    c_opt.step()
                
                metric_rew = sum(avg_rews) / len(avg_rews)
                metric_a = sum(avg_a_loss) / len(avg_a_loss)
                metric_c = sum(avg_c_loss) / len(avg_c_loss)

                print(f'Average Reward for update: {metric_rew}')
                print(f'Average Actor Loss for update: {metric_a}')
                print(f'Average Critic Loss for Update: {metric_c}')

                metric_rews.append(metric_rews)
                metric_a_loss.append(metric_a)
                metric_c_loss.append(metric_c)

        return metric_rews, metric_a_loss, metric_c_loss
                    

    def _generate_batch(self, 
                        env: gym.Env,  
                        buffer: TrajectoryBuffer,
                        gamma: float,
                        device: torch.device,
                        debug: bool):
        """
        Generates some number of trajectories in the given environment

        Parameters:
            env: The environment used to generate trajectories
            buffer: Where the trajectories will be stored
        """
        while len(buffer) < buffer.maxlen:
            prev_obs = env.reset()
            prev_obs_t = torch.Tensor(prev_obs).to(device) # Hold the previous state in a tensor for the network
    
            curr_gamma = 1

            done = False
            
            episode_obs = []
            episode_rewards = []
            episode_actions = []
            episode_done = []
            episode_next_obs = []
            episode_log_probs = []

            # Keep stepping through environment until it is completed
            while not done:
                # Generate an action from our network as well as its log probabilities
                dist = self.actor(prev_obs_t)

                actions = dist.sample()
                log_probs = dist.log_prob(actions).detach().cpu().numpy()

                # The environment is expecting numpy arrays, so we'll make it happy
                actions = actions.detach().cpu().numpy()

                obs, rew, done_t, done_e, _ = env.step(actions)
                 
                done_combined = np.logical_or(done_t, done_e)
                
                episode_obs.append(prev_obs)
                episode_rewards.append(rew)
                episode_actions.append(actions)
                episode_done.append(done_combined)
                episode_next_obs.append(obs)
                episode_log_probs.append(log_probs)

                # Set whether this environment is done or not
                done = np.sum((~done_combined).astype(float)) == 0
                
                curr_gamma *= gamma

                prev_obs = obs
                prev_obs_t = torch.Tensor(prev_obs).to(device)

                if debug:
                    env.render()
            if debug:
                env.close()
        
            rewards_to_go = self._compute_rew_to_go(episode_rewards, gamma)

            # Use multiple agents as trajectories to
            # help speed up convergence
            for obs_item, rew_item, rtg_item, act_item, done_item, next_obs_item, log_probs_item in zip(episode_obs, episode_rewards, rewards_to_go, episode_actions, episode_done, episode_next_obs, episode_log_probs):
                for i in range(len(obs_item)):
                    
                    buffer.add_item(obs_item[i, :], 
                                    act_item[i, :],
                                    rew_item[i],
                                    rtg_item[i],
                                    next_obs_item[i, :],
                                    log_probs_item[i, :],
                                    done_item[i]
                                    )

    def _compute_rew_to_go(self, episode_rewards: List[np.ndarray], gamma: float) -> List[np.ndarray]:
        """
        According to the pseudocode at https://spinningup.openai.com/en/latest/algorithms/ppo.html,
        we should compute the rewards to go, which are the discounted rewards for the episode.

        This involves going through the rewards backwards and computing the discounted reward at each episode
        """
        prev_reward = 0
        rewards_to_go = []

        for episode_reward in reversed(episode_rewards):
            prev_reward = episode_reward + gamma * prev_reward
            rewards_to_go.append(prev_reward)
        
        # Need to reverse this list so it lines up with the episodes
        rewards_to_go.reverse()

        return rewards_to_go
