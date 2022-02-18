from collections import deque
import random

# The replay buffer stores the experiences of the agent during training, and then
# randomwly sample experiences to use for learning - experience replay

#because DDPG is an off-policy algorithm, the replay buffer can be large
# allowing the algorithm to benefit from learning across a set of uncorrelated
# transitions
class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    # batch_size specifies the number of experiences to add
    # to the batch. if the replay buffer has < batch_size
    # return all elements.
    def getBatch(self, batch_size):
        # Randomly sample batch_size examples

        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            # when the replay buffer is full the old samples are discarded
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0