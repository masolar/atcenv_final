import numpy as np

class ReplayBuffer(object):
    def __init__(self, buffer_capacity, batch_size, min_size_buffer, n_agents, state_size, action_size):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.min_size_buffer = min_size_buffer
        self.buffer_counter = 0
        self.n_games = 0
        self.n_agents = n_agents
        self.list_actors_dimension = np.ones(n_agents) * state_size
        self.critic_dimension = int(sum(self.list_actors_dimension))
        self.list_actor_n_actions = np.ones(n_agents) * action_size
        
        self.states = np.zeros((self.buffer_capacity, self.critic_dimension))
        self.rewards = np.zeros((self.buffer_capacity, self.n_agents))
        self.next_states = np.zeros((self.buffer_capacity, self.critic_dimension))
        self.dones = np.zeros((self.buffer_capacity, self.n_agents), dtype=bool)

        self.list_actors_states = []
        self.list_actors_next_states = []
        self.list_actors_actions = []
        
        for n in range(self.n_agents):
            self.list_actors_states.append(np.zeros((self.buffer_capacity, int(self.list_actors_dimension[n]))))
            self.list_actors_next_states.append(np.zeros((self.buffer_capacity, int(self.list_actors_dimension[n]))))
            self.list_actors_actions.append(np.zeros((self.buffer_capacity, int(self.list_actor_n_actions[n]))))
            
    def __len__(self):
        return self.buffer_counter
        
    def check_buffer_size(self):
        return self.buffer_counter >= self.batch_size and self.buffer_counter >= self.min_size_buffer
    
    def update_n_games(self):
        self.n_games += 1
          
    def add_record(self, actor_states, actor_next_states, actions, state, next_state, reward, done):
        
        index = self.buffer_counter % self.buffer_capacity

        for agent_index in range(self.n_agents):
            self.list_actors_states[agent_index][index] = actor_states[agent_index]
            self.list_actors_next_states[agent_index][index] = actor_next_states[agent_index]
            self.list_actors_actions[agent_index][index] = actions[agent_index]

        self.states[index] = state
        self.next_states[index] = next_state
        self.rewards[index] = reward
        self.dones[index] = done
            
        self.buffer_counter += 1
            
    def get_minibatch(self):
        # If the counter is less than the capacity we don't want to take zeros records, 
        # if the cunter is higher we don't access the record using the counter 
        # because older records are deleted to make space for new one
        buffer_range = min(self.buffer_counter, self.buffer_capacity)

        batch_index = np.random.choice(buffer_range, self.batch_size, replace=False)

        # Take indices
        state = self.states[batch_index]
        reward = self.rewards[batch_index]
        next_state = self.next_states[batch_index]
        done = self.dones[batch_index]
            
        actors_state = [self.list_actors_states[index][batch_index] for index in range(self.n_agents)]
        actors_next_state = [self.list_actors_next_states[index][batch_index] for index in range(self.n_agents)]
        actors_action = [self.list_actors_actions[index][batch_index] for index in range(self.n_agents)]

        return state, reward, next_state, done, actors_state, actors_next_state, actors_action
    