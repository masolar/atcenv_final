"""
Example
"""



if __name__ == "__main__":
    import random
    random.seed(42)
    from jsonargparse import ArgumentParser, ActionConfigFile
    from atcenv import Environment
    import time
    from tqdm import tqdm

    # RL models
    from atcenv.DDPG.DDPG import DDPG
    import atcenv.DDPG.TempConfig as tc
    from atcenv.SAC.sac import SAC
    import copy
    import numpy as np

    parser = ArgumentParser(
        prog='Conflict resolution environment',
        description='Basic conflict resolution environment for training policies with reinforcement learning',
        print_config='--print_config',
        parser_mode='yaml'
    )
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--config', action=ActionConfigFile)
    parser.add_class_arguments(Environment, 'env')

    # parse arguments
    args = parser.parse_args()

    # init environment
    env = Environment(**vars(args.env))

    RL = DDPG()
    #RL = SAC()

    # increase number of flights
    rew_list = []
    state_list = []

    # run episodes
    for e in tqdm(range(args.episodes)):        
        episode_name = "EPISODE_" + str(e) 

        # reset environment
        # train with an increasing number of aircraft
        number_of_aircraft = min(int(e/100)+1,10)
        obs = env.reset(number_of_aircraft)

        # set done status to false
        done = False

        # save how many steps it took for this episode to finish
        number_steps_until_done = 0
        # save how many conflics happened in eacj episode
        number_conflicts = 0

        # execute one episode
        while not done:
            # get actions from RL model
            actions = []
            for obs_i in obs:
                #actions.append(RL.do_step(obs_i, episode_name, env.max_speed, env.min_speed))
                actions.append([0,0])

            obs0 = copy.deepcopy(obs)

            # perform step with dummy action
            obs, rew, done, info = env.step(actions)
            for rew_i in rew:
                rew_list.append(rew_i)
            for obs_i in obs:
                state_list.append(obs_i)
            
            # train the RL model
            for it_obs in range(len(obs)):
                RL.setResult(episode_name, obs0[it_obs], obs[it_obs], rew[it_obs], actions[it_obs], done, env.max_speed, env.min_speed)

            # comment render out for faster processing
            env.render()
            number_steps_until_done += 1
            number_conflicts += sum(env.conflicts)
            time.sleep(0.05)

        # save information
        #RL.update() # train the model
        RL.episode_end(episode_name)
        tc.dump_pickle(number_steps_until_done, 'results/save/numbersteps_' + episode_name)
        tc.dump_pickle(number_conflicts, 'results/save/numberconflicts_' + episode_name)
        print(episode_name,'ended in', number_steps_until_done, 'runs, with', number_conflicts, 'conflicts, number of aircraft=', number_of_aircraft)        
        np.savetxt('rewards.csv', rew_list)
        np.savetxt('states.csv', state_list)
        # close rendering
        env.close()
