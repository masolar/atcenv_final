import numpy as np
import tensorflow as tf
from  atcenv.MADDDPG.replay_buffer import ReplayBuffer
from  atcenv.MADDDPG.agent import Agent
from atcenv.MADDDPG.replay_buffer import ReplayBuffer
from tensorflow.keras import optimizers as opt

BUFFER_SIZE = 100000
BATCH_SIZE = 256
TAU = 0.001  # Target Network HyperParameters, for soft update of target parameters
LRA = 0.0001  # Learning rate for Actor
LRC = 0.001  # Lerning rate for Critic
EPSILON = 0.1  
ALPHA = 0.9  # learning rate
GAMMA = 0.99


class SuperAgent:
    def __init__(self, n_agents, state_size, action_size):
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, BATCH_SIZE, n_agents, state_size, action_size)
        self.n_agents = n_agents
        self.agents = [Agent(i, LRA, LRC, GAMMA, TAU, state_size, action_size, n_agents) for i in range(self.n_agents)]
        self.action_size = action_size
        self.counter = 1
        
    def get_actions(self, agents_states):
        list_actions = [self.agents[index].get_actions(agents_states[index][None, :]) for index in range(self.n_agents)]
        list_actions = [list_actions[index].numpy()[0] for index in range(self.n_agents)]
        return list_actions
    
    def episode_end(self, scenarioName):
        for agent in self.agents:
            agent.save(scenarioName)            
    
    def load(self):
        for agent in self.agents:
            agent.load()       

    def update_batch(self,state, nextstate, actions, all_state, all_nextstate, reward, done):
        # Store transition in the replay buffer.

        self.replay_buffer.add_record(state, nextstate, actions, all_state, all_nextstate, reward, done)
        if self.replay_buffer.check_buffer_size() and self.counter % 10 == 0:
            self.train() 
            self.counter = 1
        self.counter += 1      
    
    def train(self):        
        state, reward, next_state, done, actors_state, actors_next_state, actors_action = self.replay_buffer.get_minibatch()
        
        states = tf.convert_to_tensor(state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_state, dtype=tf.float32)

        actors_states = [tf.convert_to_tensor(s, dtype=tf.float32) for s in actors_state]
        actors_next_states = [tf.convert_to_tensor(s, dtype=tf.float32) for s in actors_next_state]
        actors_actions = [tf.convert_to_tensor(s, dtype=tf.float32) for s in actors_action]
        
        with tf.GradientTape(persistent=True) as tape:
            target_actions = [self.agents[index].target_actor(actors_next_states[index]) for index in range(self.n_agents)]
            policy_actions = [self.agents[index].actor(actors_states[index]) for index in range(self.n_agents)]
            
            concat_target_actions = tf.concat(target_actions, axis=1)
            concat_policy_actions = tf.concat(policy_actions, axis=1)
            concat_actors_action = tf.concat(actors_actions, axis=1)
            
            target_critic_values = [tf.squeeze(self.agents[index].target_critic(next_states, concat_target_actions), 1) for index in range(self.n_agents)]
            critic_values = [tf.squeeze(self.agents[index].critic(states, concat_actors_action), 1) for index in range(self.n_agents)]
            targets = [rewards[:, index] + self.agents[index].gamma * target_critic_values[index] * (1-done[:, index]) for index in range(self.n_agents)]
            critic_losses = [tf.keras.losses.MSE(targets[index], critic_values[index]) for index in range(self.n_agents)]
            
            actor_losses = [-self.agents[index].critic(states, concat_policy_actions) for index in range(self.n_agents)]
            actor_losses = [tf.math.reduce_mean(actor_losses[index]) for index in range(self.n_agents)]
        
        critic_gradients = [tape.gradient(critic_losses[index], self.agents[index].critic.trainable_variables) for index in range(self.n_agents)]
        actor_gradients = [tape.gradient(actor_losses[index], self.agents[index].actor.trainable_variables) for index in range(self.n_agents)]
        
        for index in range(self.n_agents):
            self.agents[index].critic.optimizer.apply_gradients(zip(critic_gradients[index], self.agents[index].critic.trainable_variables))
            self.agents[index].actor.optimizer.apply_gradients(zip(actor_gradients[index], self.agents[index].actor.trainable_variables))
            self.agents[index].update_target_networks(self.agents[index].tau)