"""
Example
"""

import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from IPython.display import clear_output
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
    parser.add_argument('--episodes', type=int, default=100)
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
    tot_rew_list = []
    total_step =0
    # beginning of the algorithm
    for e in tqdm(range(100)):
        Num_Train = "Num_Train_" + str(e)
        step = 0
        memory = deque()
        episodes = 0
        # update model after 2048 steps
        while episodes < 20:
            episodes += 1
            number_of_aircraft = 10  # min(int(e/500)+5,10)
            obs = env.reset(number_of_aircraft)
            for obs_i in obs:
                RL.normalizeState(obs_i, env.max_speed, env.min_speed)
            done = False
            # save how many conflics happened in eacj episode
            number_conflicts = 0
            # save different from optimal speed
            average_speed_dif = 0
            tot_rew = 0
            # start sampling trajactories
            while not done:
                step += 1
                total_step += 1
                actions = RL.do_step(obs, env.max_speed, env.min_speed, test=test)
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
                # save transition
                memory.append([obs0, actions, rew, mask])
                tot_rew += rew
                while len(obs) < len(obs0):
                    obs.append( [0] * 14) # STATE_SIZE = 14
                number_conflicts += len(env.conflicts)
                average_speed_dif = np.average([env.average_speed_dif, average_speed_dif])
            # in one episode, the average total reward of every plane
            tot_rew_list.append(sum(tot_rew)/number_of_aircraft)
            conf_list.append(number_conflicts)
            speeddif_list.append(average_speed_dif)

            if total_step % 1000 == 0:
                clear_output(wait=True)
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4), dpi=100)
                ax1.plot(tot_rew_list)
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('Total Reward')
                ax1.set_title('Average Total Rewards of Planes')
                ax1.grid()

                ax2.plot(conf_list)
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Conflicts')
                ax2.set_title('Conflicts Per Episode')
                ax2.grid()

                ax3.plot(speeddif_list)
                ax3.set_xlabel('Episode')
                ax3.set_ylabel('Average Speed Difference')
                ax3.set_title('Average Speed Dif Per Episode')
                ax3.grid()

                plt.show()

        if e%100 == 0 and not test:
            RL.save_models(e)
        if e % 10 == 0:
            env.render()
            time.sleep(2)
        print(f'Done aircraft: {len(env.done)}')  
        print(f'Done aircraft IDs: {env.done}')
        print(Num_Train,'ended in', step, 'runs, with', np.mean(np.array(conf_list)), 'conflicts (rolling av100), reward (rolling av100)=', np.mean(np.array(tot_rew_list)), 'speed dif (rolling av100)', np.mean(np.array(average_speed_dif)))
        RL.train(memory)
    np.savetxt('./results/data/total_reward.csv', np.array(tot_rew_list), delimiter=',')
    np.savetxt('./results/data/conflicts.csv', np.array(conf_list), delimiter=',')
    np.savetxt('./results/data/speed_dif.csv', np.array(speeddif_list), delimiter=',')
