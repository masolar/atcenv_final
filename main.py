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
    from atcenv.SAC.sac import SAC
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

    # init environment
    env = Environment(**vars(args.env))

    RL = DDPG()
    #RL = SAC()

    # run episodes
    for e in tqdm(range(args.episodes)):
        # reset environment
        obs = env.reset()

        # set done status to false
        done = False

        # execute one episode
        while not done:
            episode_name = "EPISODE_" + str(e) 

            # get actions from RL model
            actions = []
            for obs_i in obs:
                actions.append(RL.do_step(obs_i, episode_name))

            obs0 = copy.deepcopy(obs)

            # perform step with dummy action
            obs, rew, done, info = env.step(actions)

            # train the RL model
            for it_obs in range(len(obs)):
                RL.setResult(episode_name, obs0[it_obs], obs[it_obs], rew[it_obs], actions[it_obs], done)

            env.render()
            #time.sleep(0.05)

        # close rendering
        env.close()
