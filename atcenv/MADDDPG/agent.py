import numpy as np
import tensorflow as tf
from atcenv.MADDDPG.networks import Actor, Critic
import atcenv.MADDDPG.TempConfig as tc

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

        tf.compat.v1.disable_eager_execution()
        sess = tf.compat.v1.Session()
        tf.compat.v1.keras.backend.set_session(sess)
        
        self.actor = Actor("actor_" + self.agent_name, sess, state_size, action_size)
        self.critic = Critic("critic_" + self.agent_name,  sess, state_size*n_agents, action_size*n_agents)        
        
    def get_actions(self, actor_states):
        actions = self.actor.model.predict(actor_states.reshape(1, actor_states.shape[0]))
        actions = actions[0] 
        actions[0] = np.clip(actions[0] + np.random.uniform(-1,1), -1, 1) # tanh activation function
        actions[1] = np.clip(actions[1] + np.random.uniform(-1,1), 0, 1) # sigmoid activation function

        return actions
    
    def add_loss(self, loss):
        self.L = np.append(self.L, loss)
    
    def save(self, scenarioName):
        repetition = int(scenarioName.split('EPISODE_')[1])
        if repetition % 500 == 0:
            tc.save_DDQL('results', "DDPG_critic_" + scenarioName + "_" + self.critic.net_name + ".h5", self.critic)
            tc.save_DDQL('results', "DDPG_actor_" + scenarioName + "_" + self.actor.net_name + ".h5", self.actor)
            tc.dump_pickle(self.L, 'results/save/loss_' + scenarioName)
        
    def load(self):
        self.actor.model.load_weights(self.allWeights_actor[0])
        self.critic.model.load_weights(self.allWeights_critic[0])

