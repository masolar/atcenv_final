import numpy as np
import tensorflow as tf
from  atcenv.MADDDPG.replay_buffer import ReplayBuffer
from  atcenv.MADDDPG.agent import Agent
from atcenv.MADDDPG.replay_buffer import ReplayBuffer

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
        list_actions = [self.agents[index].get_actions(agents_states[index]) for index in range(self.n_agents)]
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

        # join all actors action for each step
        all_actors_action = [[] for _ in range(self.replay_buffer.batch_size)]
        for index2 in range(self.replay_buffer.batch_size):   
            all_actors_action[index2] = [actors_action[index][index2] for index in range(self.n_agents)]
            all_actors_action[index2] = np.concatenate(all_actors_action[index2] )

        new_actions_predicted = [[] for _ in range(self.n_agents)]
        target_q_values = [[] for _ in range(self.n_agents)]
        loss = [[] for _ in range(self.n_agents)]
        a_for_grad = [[] for _ in range(self.n_agents)]
        for index in range(self.n_agents):
            # calculate targets
            new_actions_predicted[index] = self.agents[index].actor.target_model.predict(actors_next_state[index])

        # join actions
        joint_new_actions_predicted = [[] for _ in range(self.replay_buffer.batch_size)]
        for index2 in range(self.replay_buffer.batch_size):   
            joint_new_actions_predicted[index2] = [new_actions_predicted[index][index2] for index in range(self.n_agents)]
            joint_new_actions_predicted[index2] = np.concatenate(joint_new_actions_predicted[index2])
        
        for index in range(self.n_agents):
            target_q_values[index] = self.agents[index].critic.target_model.predict([next_state, np.array(joint_new_actions_predicted)])

        y_t = [[] for _ in range(self.n_agents)]
        for index in range(self.n_agents):   
            if np.any(done[:, index]): # done are all equal
                y_t[index] = reward[:, index]
            else:
                target_q_values[index] *= GAMMA 
                for it in range(len(target_q_values[index])):
                    target_q_values[index][it] += reward[:, index][it]
                y_t[index] = target_q_values[index]  

        for index in range(self.n_agents):
            loss[index] = self.agents[index].critic.model.train_on_batch([state, np.array(all_actors_action)], np.array(y_t[index]))            
            self.agents[index].add_loss(loss[index])
            a_for_grad[index] = self.agents[index].actor.model.predict(actors_state[index])
        
        grads = [[] for _ in range(self.n_agents)]
        aux_a_for_grad = [[] for _ in range(self.replay_buffer.batch_size)]
        for index2 in range(self.replay_buffer.batch_size):  
            aux_a_for_grad[index2] =  [a_for_grad[index][index2] for index in range(self.n_agents)]
            aux_a_for_grad[index2] = np.concatenate(aux_a_for_grad[index2])

        for index in range(self.n_agents):
            grads[index] = self.agents[index].critic.gradients(state, aux_a_for_grad)  

        for index in range(self.n_agents):      
            aux_grad = [[] for _ in range(self.replay_buffer.batch_size)]
            for index2 in range(self.replay_buffer.batch_size):  
                aux_grad[index2] = grads[index][index2][index*self.action_size:index*self.action_size+self.action_size]
            self.agents[index].actor.train(actors_state[index], aux_grad)
            self.agents[index].actor.target_train()
            self.agents[index].critic.target_train()