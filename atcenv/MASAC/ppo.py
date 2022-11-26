from math import gamma
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from atcenv.MASAC.models import Actor, Critic
from torch.nn.utils.clip_grad import clip_grad_norm_


GAMMMA = 0.99
lambd = 0.98
TAU =5e-3
INITIAL_RANDOM_STEPS = 100
POLICY_UPDATE_FREQUENCE = 2
NUM_AGENTS = 10

BUFFER_SIZE = 1000000
BATCH_SIZE = 32

ACTION_DIM = 2
STATE_DIM = 14
NUMBER_INTRUDERS_STATE = 2
epsilon = 0.2
MEANS = [57000,57000,0,0,0,0,0,0]
STDS = [31500,31500,100000,100000,1,1,1,1]

class MaSacAgent:
    def __init__(self):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print('DEVICE USED: ', torch.cuda.device(torch.cuda.current_device()), torch.cuda.get_device_name(0))
    
        except:
            # Cuda isn't available
            self.device = torch.device("cpu")
            print('DEVICE USED: CPU')
        
        self.target_alpha = -np.prod((ACTION_DIM,)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        self.actor = Actor(STATE_DIM, ACTION_DIM).to(self.device)
        self.critic = Critic(STATE_DIM)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        self.critic_loss_func = torch.nn.MSELoss()
        self.transition = [[] for i in range(NUM_AGENTS)]

        self.total_step = 0

        self.is_test = False
        
    def do_step(self, state, max_speed, min_speed, test = False, batch = False):

        if not test and self.total_step < INITIAL_RANDOM_STEPS and not self.is_test:
            selected_action = np.random.uniform(-1, 1, (len(state), ACTION_DIM))
        else:
            selected_action = []
            for i in range(len(state)):
                action = self.actor(torch.FloatTensor(state[i]).to(self.device))[0].detach().cpu().numpy()
                selected_action.append(action)
            selected_action = np.array(selected_action)
            selected_action = np.clip(selected_action, -1, 1)

        self.total_step += 1
        return selected_action.tolist()


    def train(self, memory):
        memory = np.array(memory)
        states = torch.tensor(list(memory[:, 0]), dtype=torch.float32)
        actions = torch.tensor(list(memory[:, 1]), dtype=torch.float32)
        rewards = torch.tensor(list(memory[:, 2]), dtype=torch.float32)
        masks = torch.tensor(list(memory[:, 3]), dtype=torch.float32)
        values = self.critic(states)
        values = torch.reshape(values, (-1,))

        returns, advants = self.get_gae(rewards, masks, values)
        old_actions, old_log_prob = self.actor(states)
        n = len(states)
        arr = np.arange(n)
        for epoch in range(1):
            # np.random.shuffle(arr)
            for i in range(n // (BATCH_SIZE)):
                b_index = arr[BATCH_SIZE * i:BATCH_SIZE * (i + 1)]
                b_states = states[b_index]
                b_advants = advants[b_index].unsqueeze(1)
                # b_actions = actions[b_index]
                b_returns = returns[b_index].unsqueeze(1)
                actions, log_prob = self.actor(b_states)
                old_prob = old_log_prob[b_index].detach()
                ratio = torch.exp(log_prob - old_prob)
                surrogate_loss = ratio * b_advants
                values = self.critic(b_states)
                critic_loss = self.critic_loss_func(values, b_returns)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
                clipped_loss = ratio * b_advants
                actor_loss = -torch.min(surrogate_loss, clipped_loss).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()


    def get_gae(self, rewards, masks, values):
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)
        returns = torch.zeros_like(rewards)
        advants = torch.zeros_like(rewards)
        running_returns = 0
        previous_value = 0
        running_advants = 0
        for t in reversed(range(0, len(rewards))):
            # 计算A_t并进行加权求和
            running_returns = rewards[t] + GAMMMA * running_returns * masks[t]
            running_tderror = rewards[t] + GAMMMA * previous_value * masks[t] - \
                              values.data[t]
            running_advants = running_tderror + GAMMMA * lambd * \
                              running_advants * masks[t]
            returns[t] = running_returns
            previous_value = values.data[t]
            advants[t] = running_advants
        # advants的归一化
        advants = (advants - advants.mean()) / advants.std()
        return returns, advants
    
    def save_models(self):
        torch.save(self.actor.state_dict(), "results/actor.pt")
        torch.save(self.critic.state_dict(), "results/critic.pt")

    def load_models(self):
        # The models were trained on a CUDA device
        # If you are running on a CPU-only machine, use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.
        self.actor.load_state_dict(torch.load("results/actor.pt", map_location=torch.device('cpu')))
        self.critic.load_state_dict(torch.load("results/critic.pt", map_location=torch.device('cpu')))


    def normalizeState(self, s_t, max_speed, min_speed):
         # distance to closest #NUMBER_INTRUDERS_STATE intruders
        for i in range(0, NUMBER_INTRUDERS_STATE):
            s_t[i] = (s_t[i]-MEANS[0])/(STDS[0]*2)

        # relative bearing to closest #NUMBER_INTRUDERS_STATE intruders
        for i in range(NUMBER_INTRUDERS_STATE, NUMBER_INTRUDERS_STATE*2):
            s_t[i] = (s_t[i]-MEANS[1])/(STDS[1]*2)

        for i in range(NUMBER_INTRUDERS_STATE*2, NUMBER_INTRUDERS_STATE*3):
            s_t[i] = (s_t[i]-MEANS[2])/(STDS[2]*2)

        for i in range(NUMBER_INTRUDERS_STATE*3, NUMBER_INTRUDERS_STATE*4):
            s_t[i] = (s_t[i]-MEANS[3])/(STDS[3]*2)

        for i in range(NUMBER_INTRUDERS_STATE*4, NUMBER_INTRUDERS_STATE*5):
            s_t[i] = (s_t[i])/(3.1415)

        # current bearing

        # current speed
        s_t[NUMBER_INTRUDERS_STATE*5] = ((s_t[NUMBER_INTRUDERS_STATE*5]-min_speed)/(max_speed-min_speed))*2 - 1
        # optimal speed
        s_t[NUMBER_INTRUDERS_STATE*5 + 1] = ((s_t[NUMBER_INTRUDERS_STATE*5 + 1]-min_speed)/(max_speed-min_speed))*2 - 1
        # # distance to target
        # s_t[NUMBER_INTRUDERS_STATE*2 + 2] = s_t[NUMBER_INTRUDERS_STATE*2 + 2]/MAX_DISTANCE
        # # bearing to target
        s_t[NUMBER_INTRUDERS_STATE*5+2] = s_t[NUMBER_INTRUDERS_STATE*5+2]
        s_t[NUMBER_INTRUDERS_STATE*5+3] = s_t[NUMBER_INTRUDERS_STATE*5+3]

        # s_t[0] = s_t[0]/MAX_BEARING

        return s_t