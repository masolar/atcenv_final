import numpy as np
import tensorflow as tf
from atcenv.MADDDPG.networks import Actor, Critic
import atcenv.MADDDPG.TempConfig as tc
from tensorflow.keras import optimizers as opt

GAMMA = 0.99

class Agent:
    def __init__(self, i, actor_lr, critic_lr, gamma, tau, state_size, action_size,n_agents):
        
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        
        self.actor_dims= state_size
        self.n_actions = action_size
        self.L = np.array([])
        
        self.agent_name = "agent_number_{}".format(i)

        self.actor = Actor("actor_" + self.agent_name)
        self.critic = Critic("critic_" + self.agent_name)
        self.target_actor = Actor("target_actor_" + self.agent_name)
        self.target_critic = Critic("target_critic_" + self.agent_name)
        
        self.actor.compile(optimizer=opt.Adam(learning_rate=actor_lr))
        self.critic.compile(optimizer=opt.Adam(learning_rate=critic_lr))
        self.target_actor.compile(optimizer=opt.Adam(learning_rate=actor_lr))
        self.target_critic.compile(optimizer=opt.Adam(learning_rate=critic_lr))
        
        actor_weights = self.actor.get_weights()
        critic_weights = self.critic.get_weights()
        
        self.target_actor.set_weights(actor_weights)
        self.target_critic.set_weights(critic_weights)
        
    def get_actions(self, actor_states):
        actions = self.actor(actor_states)
        return actions
    
    def add_loss(self, loss):
        self.L = np.append(self.L, loss)
    
    def save(self, scenarioName):
        repetition = int(scenarioName.split('EPISODE_')[1])
        if repetition % 500 == 0:
            tc.save_DDQL('results', scenarioName + "_" + self.critic.net_name + ".h5", self.critic)
            tc.save_DDQL('results', scenarioName + "_" + self.actor.net_name + ".h5", self.actor)
            tc.save_DDQL('results', scenarioName + "_" + self.target_critic.net_name + ".h5", self.critic)
            tc.save_DDQL('results', scenarioName + "_" + self.target_actor.net_name + ".h5", self.actor)
            tc.dump_pickle(self.L, 'results/save/loss_' + scenarioName)
        
    def load(self):
        self.actor.model.load_weights(self.allWeights_actor[0])
        self.critic.model.load_weights(self.allWeights_critic[0])

    def update_target_networks(self, tau):
        actor_weights = self.actor.weights
        target_actor_weights = self.target_actor.weights
        for index in range(len(actor_weights)):
            target_actor_weights[index] = tau * actor_weights[index] + (1 - tau) * target_actor_weights[index]

        self.target_actor.set_weights(target_actor_weights)
        
        critic_weights = self.critic.weights
        target_critic_weights = self.target_critic.weights
    
        for index in range(len(critic_weights)):
            target_critic_weights[index] = tau * critic_weights[index] + (1 - tau) * target_critic_weights[index]

        self.target_critic.set_weights(target_critic_weights)