"""
Main file. 
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
    from atcenv.MADDDPG.maddpg import MADDPG
    import atcenv.DDPG.TempConfig as tc
    from atcenv.SAC.sac import SAC
    import copy
    import numpy as np    
    from sklearn.cluster import KMeans
    from pandas import DataFrame
    from shapely.geometry import Point

    STATE_SIZE = 14
    ACTION_SIZE = 2
    NUMBER_ACTORS_MARL = 10

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

    #RL = DDPG()
    RL = MADDPG(NUMBER_ACTORS_MARL, STATE_SIZE, ACTION_SIZE)
    #RL = SAC()

    # increase number of flights
    rew_list = []
    state_list = []

    # run episodes
    for e in tqdm(range(args.episodes)):        
        episode_name = "EPISODE_" + str(e) 

        # reset environment
        # train with an increasing number of aircraft
        #number_of_aircraft = min(int(e/100)+1,10)
        number_of_aircraft = 10
        obs = env.reset(number_of_aircraft, NUMBER_ACTORS_MARL)

        # set done status to false
        done = False

        # save how many steps it took for this episode to finish
        number_steps_until_done = 0
        # save how many conflics happened in eacj episode
        number_conflicts = 0

        # execute one episode
        while not done:           
            obs0 = copy.deepcopy(obs)
            
            # get actions from RL model 
            if type(RL) is MADDPG:
                actions = [ [] for _ in range(number_of_aircraft)]
                if number_of_aircraft > NUMBER_ACTORS_MARL: # we need to divide all aircraft into groups based on their current position
                    n_cluster = int(np.ceil(number_of_aircraft/NUMBER_ACTORS_MARL))
                    ids = []
                    x = []
                    y = []
                    for flight_idx in range(env.num_flights):
                        if flight_idx not in env.done:
                            ids.append(flight_idx)
                            x.append(env.flights[flight_idx].position.x)
                            y.append(env.flights[flight_idx].position.y)
                    df = DataFrame( {'id': ids, 'x': x, 'y': y})
                    kmeans = KMeans(n_clusters = n_cluster).fit(df[['x', 'y']])    
                    cluster_indexes = [ [] for _ in range(n_cluster)]
                    # kmeans does not limit the number of points per cluster, so we have to do it ourselfs
                    for flight_idx in range(env.num_flights):
                        distance_to_centers = [0] * n_cluster
                        for idx_center in range(n_cluster):
                            distance_to_centers[idx_center] = env.flights[flight_idx].position.distance(Point(kmeans.cluster_centers_[idx_center]))
                        picked_center = 0
                        while len(cluster_indexes[np.argsort(distance_to_centers)[picked_center]]) >=NUMBER_ACTORS_MARL:
                            picked_center +=1
                        cluster_indexes[np.argsort(distance_to_centers)[picked_center]].append(flight_idx)
                    for cluster_idx in range(n_cluster):            
                        indexes = np.array(cluster_indexes[cluster_idx])
                        obs_cluster = np.array(obs)[indexes]
                        actions_aux = RL.do_step(obs_cluster, episode_name, env.max_speed, env.min_speed)
                        for index in indexes:
                            actions[index] = actions_aux.pop()
                else:
                    actions  = RL.do_step(obs, episode_name, env.max_speed, env.min_speed)
            else:
                for obs_i in obs:
                    actions = RL.do_step(obs_i, episode_name, env.max_speed, env.min_speed)

            # perform step with dummy action
            obs, rew, done, info = env.step(actions, type(RL) is MADDPG, NUMBER_ACTORS_MARL)
            for rew_i in rew:
                rew_list.append(rew_i)
            for obs_i in obs:
                state_list.append(obs_i)
            
            # train the RL model
            # comment out on testing
            if type(RL) is MADDPG:
                if number_of_aircraft > NUMBER_ACTORS_MARL:
                     for clusters_idx in range(n_cluster):
                        indexes = np.array(cluster_indexes[cluster_idx])
                        obs0_cluster = np.array(obs0)[indexes]
                        obs_cluster  =  np.array(obs)[indexes]
                        actions_cluster =  np.array(actions)[indexes]
                        rew_cluster = sum(rew[indexes])
                        RL.setResult(episode_name, obs0_cluster, obs_cluster, rew_cluster, actions_cluster, done, env.max_speed, env.min_speed)
                else:
                    rew = sum(rew)
                    RL.setResult(episode_name, obs0, obs, rew, actions, done, env.max_speed, env.min_speed)
            else:
                for it_obs in range(len(obs)):
                    RL.setResult(episode_name, obs0[it_obs], obs[it_obs], rew[it_obs], actions[it_obs], done, env.max_speed, env.min_speed)

            # comment render out for faster processing
            env.render()
            number_steps_until_done += 1
            number_conflicts += sum(env.conflicts)
            #time.sleep(0.05)

        # save information
        #RL.update() # train the model
        # comment out on testing
        RL.episode_end(episode_name)
        tc.dump_pickle(number_steps_until_done, 'results/save/numbersteps_' + episode_name)
        tc.dump_pickle(number_conflicts, 'results/save/numberconflicts_' + episode_name)
        print(episode_name,'ended in', number_steps_until_done, 'runs, with', number_conflicts, 'conflicts, number of aircraft=', number_of_aircraft)        
        np.savetxt('rewards.csv', rew_list)
        np.savetxt('states.csv', state_list)
        # close rendering
        env.close()
