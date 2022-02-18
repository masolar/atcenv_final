from atcenv.DDPG.actor import ActorNetwork
from atcenv.DDPG.critic import CriticNetwork
from atcenv.DDPG.OU import OrnsteinUhlenbeckActionNoise
from atcenv.DDPG.ReplayBuffer import ReplayBuffer
import atcenv.DDPG.TempConfig as tc
import tensorflow as tf
import numpy as np


BUFFER_SIZE = 1000000
BATCH_SIZE = 256
TAU = 0.001  # Target Network HyperParameters, for soft update of target parameters
LRA = 0.0001  # Learning rate for Actor
LRC = 0.001  # Lerning rate for Critic
EPSILON = 0.1  
ALPHA = 0.9  # learning rate
GAMMA = 0.99

ACTION_DIM = 2
STATE_DIM = 15


# NORMALIZATION_FACTOR = 150  # value achieved by tuning


class DDPG(object):

    def __init__(self):

        # save performed actions
        self.actions = dict()
        self.reward_per_action = dict()
        self.L = np.array([])

        tf.compat.v1.disable_eager_execution()
        sess = tf.compat.v1.Session()
        tf.compat.v1.keras.backend.set_session(sess)

        self.action_dim = ACTION_DIM
        self.state_dim = STATE_DIM
  
        self.actor = ActorNetwork(sess, self.state_dim, self.action_dim, BATCH_SIZE, TAU, LRA)
        self.critic = CriticNetwork(sess, self.state_dim, self.action_dim, BATCH_SIZE, TAU, LRC)

        # Initialize replay memorybatch
        self.buff = ReplayBuffer(BUFFER_SIZE) 
        # exploration noise
        self.noise1 = OrnsteinUhlenbeckActionNoise(ACTION_DIM)

    def normalizeState(self, s_t):
        #TODO

        return s_t

    def do_step(self, s_t, scenname):
        s_t = self.normalizeState(np.asarray(s_t))

        if s_t.shape[0] > self.state_dim:
            a_t_original = self.actor.model.predict(s_t)
        else:
            a_t_original = self.actor.model.predict(s_t.reshape(1, s_t.shape[0]))

        actions = a_t_original[0]
        #actions += self.noise1.sample()
        actions = np.clip(actions, 0, 1)

        return actions

    def batch_update(self):
        # Do the batch update
        batch = self.buff.getBatch(BATCH_SIZE)
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])
        y_t = np.asarray([e[1] for e in batch])

        # calculate targets
        if len(states) > 0:
            new_states_predicted = self.actor.target_model.predict(new_states)
            target_q_values = self.critic.target_model.predict([new_states, new_states_predicted])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]

            # update the critic given the targets
            loss = self.critic.model.train_on_batch([states, actions], y_t)
            self.L = np.append(self.L, loss)

            # update the actor policy using the sampled gradient
            a_for_grad = self.actor.model.predict(states)
            # states = states.reshape(len(states), self.state_dim)

            grads = self.critic.gradients(states, a_for_grad)
            self.actor.train(states, grads)

            self.actor.target_train()
            self.critic.target_train()

    def setResult(self, scenname, state, nextstate, rewards, actions, done):

        if self.actions.get(scenname) is None:
            self.actions[scenname] = np.array([])
        if self.reward_per_action.get(scenname) is None:
            self.reward_per_action[scenname] = np.array([])
  
        self.reward_per_action[scenname] = np.append(self.reward_per_action[scenname], rewards)
        self.actions[scenname] = np.append(self.actions[scenname], actions)

        state = self.normalizeState(np.asarray(state))
        nextstate = self.normalizeState(np.asarray(nextstate))

        self.buff.add(state, actions, rewards, nextstate, done)  # Add replay buffer
        self.batch_update()

        if done:
            self.episode_end()

    def episode_end(self, scenarioName):
        print('episode end', scenarioName)

        repetition = scenarioName.split('Rep')[1]
        repetition = int(repetition.split('.')[0])
        if repetition % 5 == 0:
            tc.save_DDQL('results', "DDPG_critic_" + scenarioName + ".h5", self.critic)
            tc.save_DDQL('results', "DDPG_actor_" + scenarioName + ".h5", self.actor)
            tc.dump_pickle(self.L, 'results/save/loss_' + scenarioName)

        if scenarioName not in self.reward_per_action:
            return

        tc.dump_pickle(self.reward_per_action[scenarioName], 'results/save/reward_' + scenarioName)
        tc.dump_pickle(self.actions[scenarioName], 'results/save/actions_' + scenarioName)