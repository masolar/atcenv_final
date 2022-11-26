"""
Example
"""

import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import tracemalloc

if __name__ == "__main__":
    import random
    random.seed(52)
    from jsonargparse import ArgumentParser, ActionConfigFile
    from atcenv import Environment
    import time
    from tqdm import tqdm

    # RL model
    import atcenv.TempConfig as tc
    from atcenv.MASAC.ppo import MaSacAgent
    import copy
    parser = ArgumentParser(
        prog='Conflict resolution environment',
        description='Basic conflict resolution environment for training policies with reinforcement learning',
        print_config='--print_config',
        parser_mode='yaml'
    )
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--config', action=ActionConfigFile)
    parser.add_class_arguments(Environment, 'env')

    # parse arguments
    args = parser.parse_args()
    #tracemalloc.start()

    # init environment
    env = Environment(**vars(args.env))

    #RL = DDPG()
    RL = MaSacAgent()

    load_models = False
    test = False

    if load_models:
        RL.load_models()
    # increase number of flights
    conf_list = []
    speeddif_list = []
    # run episodes
    state_list = []
    episodes = 0
    avg_rewards = []
    show_episodes = []

    # beginning of the algorithm
    for e in tqdm(range(args.episodes)):
        episode_name = "EPISODE_" + str(e)
        step = 0
        memory = deque()
        tot_rew_list = []
        while step < 2048:
            episodes += 1
            number_of_aircraft = 10  # min(int(e/500)+5,10)
            obs = env.reset(number_of_aircraft)
            for obs_i in obs:
                RL.normalizeState(obs_i, env.max_speed, env.min_speed)
            done = False
            number_steps_until_done = 0
            # save how many conflics happened in eacj episode
            number_conflicts = 0
            # save different from optimal speed
            average_speed_dif = 0

            tot_rew = 0
            for _ in range(10000):
                #for obs_i in obs:
                # print(obs_i)
                step += 1
                actions = RL.do_step(obs, env.max_speed, env.min_speed, test=test)
                    # actions.append((np.random.rand(2)-0.5)*2)
                    #actions.append([0,0])

                obs0 = copy.deepcopy(obs)

                # perform step with dummy action
                obs, rew, done_t, done_e, info = env.step(actions)

                for obs_i in obs:
                   RL.normalizeState(obs_i, env.max_speed, env.min_speed)

                if done_t or done_e:
                    print(len(memory))
                    done = True
                    break
                mask = (1 - done) * 1
                memory.append([obs0, actions, rew, mask]) # rollout

                tot_rew += rew

                while len(obs) < len(obs0):
                    obs.append( [0] * 14) # STATE_SIZE = 14
                # RL.setResult(episode_name, obs0, obs, sum(rew), actions, done_e)
                    # print('obs0,',obs0[it_obs],'obs,',obs[it_obs],'done_e,', done_e)
                # comment render out for faster processing
                # if e%10 == 0:
                #     env.render()
                    #time.sleep(0.01)
                number_steps_until_done += 1
                number_conflicts += len(env.conflicts)
                average_speed_dif = np.average([env.average_speed_dif, average_speed_dif])
            tot_rew_list.append(tot_rew)
            if step >= 2048:
                show_episodes.append(episodes)
                rew_avg = np.mean(tot_rew_list)
                avg_rewards.append(rew_avg)
                plt.clf()
                plt.xlabel('Episodes')
                plt.ylabel('Rewards')
                plt.plot(show_episodes, avg_rewards, color='skyblue', label='Current')
                # plt.plot(self.avg_rewards, color='red', label='Average')
                plt.legend()
                # plt.savefig('Train.jpg')
                plt.show()
                plt.pause(0.001)
        if len(tot_rew_list) < 100:
            tot_rew_list.append(sum(tot_rew)/number_of_aircraft)
            conf_list.append(number_conflicts)
            speeddif_list.append(average_speed_dif)
        else:
            tot_rew_list[e%100 -1] = sum(tot_rew)/number_of_aircraft
            conf_list[e%100 -1] = number_conflicts
            speeddif_list[e%100 -1] = average_speed_dif
        # save information
        # if not test:
        #     RL.learn() # train the model
        # if e%100 == 0 and not test:
        #     RL.save_models()
        #RL.episode_end(episode_name)
        #np.savetxt('states.csv', state_list)
        tc.dump_pickle(number_steps_until_done, 'results/save/numbersteps_' + episode_name)
        tc.dump_pickle(number_conflicts, 'results/save/numberconflicts_' + episode_name)
        tc.dump_pickle(average_speed_dif, 'results/save/speeddif_' + episode_name)
        print(f'Done aircraft: {len(env.done)}')  
        print(f'Done aircraft IDs: {env.done}')      

        print(episode_name,'ended in', number_steps_until_done, 'runs, with', np.mean(np.array(conf_list)), 'conflicts (rolling av100), reward (rolling av100)=', np.mean(np.array(tot_rew_list)), 'speed dif (rolling av100)', np.mean(np.array(average_speed_dif)))        
        #snapshot2 = tracemalloc.take_snapshot()
        #top_stats = snapshot2.compare_to(snapshot1, 'lineno')

        #print("[ Top 10 differences ]")
        #for stat in top_stats[:10]:
        #    print(stat)
        # close rendering
        # env.close()
        RL.train(memory)
