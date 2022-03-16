import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import random_uniform

HIDDEN1_UNITS_ = 120
HIDDEN2_UNITS_ = 120
HUBER_LOSS_DELTA = 1

TAU = 0.001  # Target Network HyperParameters, for soft update of target parameters
LRA = 0.0001  # Learning rate for Actor
LRC = 0.001  # Lerning rate for Critic
EPSILON = 0.1  
ALPHA = 0.9  # learning rate
GAMMA = 0.99



class Critic(tf.keras.Model):
    def __init__(self, name,):
            
        super(Critic, self).__init__()
        
        self.hidden_0 = HIDDEN1_UNITS_
        self.hidden_1 = HIDDEN2_UNITS_

        self.net_name = name

        self.dense_0 = Dense(self.hidden_0, activation='relu')
        self.dense_1 = Dense(self.hidden_1, activation='relu')
        self.q_value = Dense(1, activation=None)
    
    def call(self, state, actors_actions):
        state_action_value = self.dense_0(tf.concat([state, actors_actions], axis=1)) # multiple actions
        state_action_value = self.dense_1(state_action_value)

        q_value = self.q_value(state_action_value)

        return q_value

class Actor(tf.keras.Model):
    def __init__(self, name, actions_dim):
        super(Actor, self).__init__()
        self.hidden_0 = HIDDEN1_UNITS_
        self.hidden_1 = HIDDEN2_UNITS_
        
        self.net_name = name

        self.dense_0 = Dense(self.hidden_0, activation='relu')
        self.dense_1 = Dense(self.hidden_1, activation='relu')
        self.policy_1 = Dense(1, activation='tanh') 
        self.policy_2 = Dense(1, activation='sigmoid') 

    def call(self, state):
        x = self.dense_0(state)
        policy = self.dense_1(x)
        policy_1 = self.policy_1(policy)
        policy_2 = self.policy_2(policy)
        policy = tf.keras.layers.concatenate([policy_1, policy_2])
        return policy