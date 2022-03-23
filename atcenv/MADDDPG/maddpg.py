import numpy as np
from atcenv.DDPG.DDPG import STATE_DIM
from atcenv.MADDDPG.super_agent import SuperAgent
import atcenv.MADDDPG.TempConfig as tc
import math
import atcenv.units as u

NUMBER_INTRUDERS_STATE = 5
MAX_DISTANCE = 100*u.nm
MAX_BEARING = math.pi

class MADDPG(object):
    def __init__(self, n_agents, state_size, action_size):      
        # save performed actions
        self.actions = dict()
        self.reward_per_action = dict()
        self.super_agent = SuperAgent(n_agents, state_size, action_size)

    def normalizeState(self, s_t, max_speed, min_speed):
        # distance to closest #NUMBER_INTRUDERS_STATE intruders
        for i in range(0, NUMBER_INTRUDERS_STATE):
            s_t[i] = min(s_t[i]/MAX_DISTANCE, 1)        

        # relative bearing to closest #NUMBER_INTRUDERS_STATE intruders
        for i in range(NUMBER_INTRUDERS_STATE, NUMBER_INTRUDERS_STATE*2):
            s_t[i] = s_t[i]/MAX_BEARING       

        # current speed
        if s_t[NUMBER_INTRUDERS_STATE*2] >= min_speed:
            s_t[NUMBER_INTRUDERS_STATE*2 ] = (s_t[NUMBER_INTRUDERS_STATE*2]-min_speed)/(max_speed-min_speed)
        # optimal speed
        if s_t[NUMBER_INTRUDERS_STATE*2 + 1] >= min_speed:
            s_t[NUMBER_INTRUDERS_STATE*2 + 1] = (s_t[NUMBER_INTRUDERS_STATE*2 + 1]-min_speed)/(max_speed-min_speed)
        # distance to target
        s_t[NUMBER_INTRUDERS_STATE*2 + 2] = s_t[NUMBER_INTRUDERS_STATE*2 + 2]/MAX_DISTANCE
        # bearing to target
        s_t[NUMBER_INTRUDERS_STATE*2 + 3] = s_t[NUMBER_INTRUDERS_STATE*2 + 3]/MAX_BEARING

        return s_t

    def do_step(self, s_t, episode_name, max_speed, min_speed):
        for it in range(len(s_t)):
            s_t[it] = self.normalizeState(np.asarray(s_t[it]), max_speed, min_speed)
        actions = self.super_agent.get_actions(s_t)
        return actions

    def setResult(self, scenname, state, nextstate, reward, actions, done, max_speed, min_speed):
        if self.actions.get(scenname) is None:
            self.actions[scenname] = np.array([])
        if self.reward_per_action.get(scenname) is None:
            self.reward_per_action[scenname] = np.array([])
  
        self.reward_per_action[scenname] = np.append(self.reward_per_action[scenname], reward)
        self.actions[scenname] = np.append(self.actions[scenname], actions)

        for it in range(len(state)):
            state[it] = self.normalizeState(np.asarray(state[it]), max_speed, min_speed)
            nextstate[it] = self.normalizeState(np.asarray(nextstate[it]), max_speed, min_speed)
        reward = reward/10

        all_state = np.concatenate(state)
        all_nextstate = np.concatenate(nextstate)

        #update
        self.super_agent.update_batch(state, nextstate, actions, all_state, all_nextstate, reward, done)

    def episode_end(self, scenarioName):
        print('episode end', scenarioName)      
        print(scenarioName, 'average reward', np.average(self.reward_per_action[scenarioName]))
        tc.dump_pickle(self.reward_per_action[scenarioName], 'results/save/reward_' + scenarioName)
        tc.dump_pickle(self.actions[scenarioName], 'results/save/actions_' + scenarioName)
        self.super_agent.episode_end(scenarioName)