
from bluesky_gym import BlueskyEnv
from ppo_network import *
from torch.optim import Adam


class ppo_trainer:
    """
    The implemetation of Proximal Policy Optimization (PPO), refer to this pseudo code to help with the understanding: 
    https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
    """

    def __init__(self, total_num_timestep, num_update_per_iteration):
        # TODO: Hyperparameter: total_num_timesteps ; num_update_per_iteration (5)
        self.total_num_timestep = total_num_timestep
        self.num_update_per_iteration = num_update_per_iteration
        pass

    def replay_memory():
        """
        Some Reading:
        - observations: (number of timesteps per batch, dimension of observation)
        - actions: (number of timesteps per batch, dimension of action)
        - log probabilities: (number of timesteps per batch)
        - rewards: (number of episodes, number of timesteps per episode)
        - reward-to-go’s: (number of timesteps per batch)
        - batch lengths: (number of episodes)
        """
        pass

    def train(self, env):
        # Initialize the gym environment
        self.env = env

        # TODO: Step 1: Initialize policy parameter, theta and value function parameters, phi (set up two network)
        self.actor = Net(self.env)
        self.critic = Net(self.env, cate="critic")

        # TODO: Step 2: Loop until total num of timestep
        for i in range(self.total_num_timestep):

            # TODO: Step 3: Collect the state, action, prev_log_probability, reward from set of trajectory by calling actor using polciy net Pi_theta_old
            # TODO: Disscussion? Sounds like a replay memory

            # TODO: Step 4: computer the rewards-to-go V_phi_k

            # TODO: Step 5: Estimate advantage value, A_k (using any method of advantage estimation) based on the current value function v_phi_k
            # A_pi_(s,a) = Q_pi_(s,a) - V_phi_k(s)

            pass

            # TODO: Loop until num_update_per_iteration
            for i in range(self.num_update_per_iteration):

                # TODO: Step 6: Update the policy by maximizing the ppo-clip objective, aka actor loss -> actor_loss = (-torch.min(l1, l2)).mean(); SGD with Adam

                # TODO: Comparing the two surrogated loss function
                # TODO: Step 6.1: l1 = (pi_theta(a_t|s_t)/pi_theta_k(a_t|s_t))*A_k -> new parameter over old parameter

                # TODO: Step 6.2: Clip the ratio  to make sure we are not stepping too far in any direction during gradient descent -> surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # TODO: Step 7: Fit value function by regression on MSE -> Adam optimizer is still quite popular
                # TODO: Step 7.1 : MSE(V,reward-to-go)

                pass
