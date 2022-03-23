"""
Environment module
"""
import gym
from typing import Dict, List
import atcenv
from atcenv.definitions import *
from gym.envs.classic_control import rendering
from shapely.geometry import LineString

import math as m
# our own packages
import numpy as np

WHITE = [255, 255, 255]
GREEN = [0, 255, 0]
BLUE = [0, 0, 255]
BLACK = [0, 0, 0]
RED = [255, 0, 0]

NUMBER_INTRUDERS_STATE = 1
MAX_DISTANCE = 250*u.nm
MAX_BEARING = math.pi

STATE_SIZE = 8
ACTION_SIZE = 2

class Environment(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self,
                 num_flights: int = 10,
                 dt: float = 5.,
                 max_area: Optional[float] = 200. * 200.,
                 min_area: Optional[float] = 125. * 125.,
                 max_speed: Optional[float] = 500.,
                 min_speed: Optional[float] = 400,
                 max_episode_len: Optional[int] = 500,
                 min_distance: Optional[float] = 5.,
                 distance_init_buffer: Optional[float] = 5.,
                 **kwargs):
        """
        Initialises the environment

        :param num_flights: numer of flights in the environment
        :param dt: time step (in seconds)
        :param max_area: maximum area of the sector (in nm^2)
        :param min_area: minimum area of the sector (in nm^2)
        :param max_speed: maximum speed of the flights (in kt)
        :param min_speed: minimum speed of the flights (in kt)
        :param max_episode_len: maximum episode length (in number of steps)
        :param min_distance: pairs of flights which distance is < min_distance are considered in conflict (in nm)
        :param distance_init_buffer: distance factor used when initialising the enviroment to avoid flights close to conflict and close to the target
        :param kwargs: other arguments of your custom environment
        """
        self.num_flights = num_flights
        self.max_area = max_area * (u.nm ** 2)
        self.min_area = min_area * (u.nm ** 2)
        self.max_speed = max_speed * u.kt
        self.min_speed = min_speed * u.kt
        self.min_distance = min_distance * u.nm
        self.max_episode_len = max_episode_len
        self.distance_init_buffer = distance_init_buffer
        self.dt = dt

        # tolerance to consider that the target has been reached (in meters)
        self.tol = self.max_speed * 1.05 * self.dt

        self.viewer = None
        self.airspace = None
        self.flights = [] # list of flights
        self.conflicts = set()  # set of flights that are in conflict
        self.done = set()  # set of flights that reached the target
        self.i = None

    def resolution(self, action: List, isMARL) -> None:
        """
        Applies the resolution actions
        If your policy can modify the speed, then remember to clip the speed of each flight
        In the range [min_speed, max_speed]
        :param action: list of resolution actions assigned to each flight
        :return:
        """

        it2 = 0      
        for i, f in enumerate(self.flights):
            if i not in self.done:
                # heading, speed
                new_track = f.track + action[it2][0] * MAX_BEARING/12
                f.track = (new_track + u.circle) % u.circle
                f.airspeed = (action[it2][1]) * (self.max_speed - self.min_speed) + self.min_speed
                if not isMARL:
                    it2 +=1
            if isMARL:
                it2 +=1 # we have a fixed number of actors, so we will still have actions even we dont use them

        # RDC: here you should implement your resolution actions
        ##########################################################
        return None
        ##########################################################

    def reward(self, isMARL) -> List:
        """
        Returns the reward assigned to each agent
        :return: reward assigned to each agent
        """
        weight_a    = -5.
        weight_b    = 2/5.
        weight_c    = -5.
        weight_d    = 0
        weight_e    = 0
        weight_f    = 0
        
        conflicts   = self.conflict_penalties() * weight_a
        drifts      = self.drift_penalties() * weight_b
        severities  = self.conflict_severity() * weight_c 
        speed_dif   = self.speedDifference() * weight_d 
        target      = self.reachedTarget() * weight_e # can also try to just ouput negative rewards
        distance_from_target = self.distanceTarget() * weight_f 
        
        tot_reward  = conflicts + severities + drifts + speed_dif + target + distance_from_target

        return tot_reward

    def distanceTarget(self):
        """
        Returns a 
        """        
        max_distance = 100*u.nm
        dist_total = np.zeros(self.num_flights)
        for i, f in enumerate(self.flights):
            if i not in self.done:
                dist_total[i] = f.position.distance(f.target)/max_distance                
                    
        return dist_total

    def reachedTarget(self):
        """
        Returns a list with aircraft that just reached the target
        :return: boolean list - 1 if aircraft have reached the target
        """        
        target = np.zeros(self.num_flights)
        for i, f in enumerate(self.flights):
            if i not in self.done:
                distance = f.position.distance(f.target)
                if distance < self.tol:
                    target[i] = 1
                    
        return target

    def speedDifference(self):
        """
        Returns a list with the diferent betwee the current aircraft and its optimal speed
        :return: float of the speed difference
        """
        speed_dif = np.zeros(self.num_flights)
        for i, f in enumerate(self.flights):
            speed_dif[i] = abs(f.airspeed - f.optimal_airspeed)/(self.max_speed - self.min_speed)
                    
        return speed_dif
        
    def conflict_penalties(self):
        """
        Returns a list with aircraft that are in conflict,
        can be used for multiplication as individual reward
        component
        :return: boolean list for conflicts
        """
        
        conflicts = np.zeros(self.num_flights)
        for i in range(self.num_flights):
            if i not in self.done:
                if i in self.conflicts:
                    conflicts[i] += 1
                    
        return conflicts
    
    def drift_penalties(self):
        """
        Returns a list with the drift angle for all aircraft,
        can be used for multiplication as individual reward
        component
        :return: float of the drift angle
        """
        
        drift = np.zeros(self.num_flights)
        for i, f in enumerate(self.flights):
            if i not in self.done:
                drift[i] = 0.5 - abs(f.drift)
        
        return drift
            
    def conflict_severity(self):
        
        severity = np.zeros(self.num_flights)
        for i in range(self.num_flights - 1):
            if i not in self.done:
                if i in self.conflicts:
                    distances = np.array([])
                    for j in list(self.conflicts - {i}):
                        distance = self.flights[i].position.distance(self.flights[j].position)
                        distances = np.append(distances,distance)
                    #conflict severity on a scale of 0-1
                    severity[i] = -1.*((min(distances)-self.min_distance)/self.min_distance)
        
        return severity

    def observation(self) -> List:
        """
        Returns the observation of each agent
        :return: observation of each agent
        """
        # observations (size = 4 * NUMBER_INTRUDERS_STATE + 4):
        # current distance to closest #NUMBER_INTRUDERS_STATE intruders
        # future distance to closest #NUMBER_INTRUDERS_STATE intruders
        # current distance in x from intruder
        # current distance in y from intruder
        # current speed
        # optimal speed
        # sin of drift error
        # cos of drift error

        observations_all = []
        cur_dis     = np.ones((self.num_flights, self.num_flights))*MAX_DISTANCE
        distance_all = np.ones((self.num_flights, self.num_flights))*MAX_DISTANCE
        bearing_all = np.ones((self.num_flights, self.num_flights))*MAX_BEARING
        dx_all  = np.ones((self.num_flights, self.num_flights))*MAX_DISTANCE
        dy_all  = np.ones((self.num_flights, self.num_flights))*MAX_DISTANCE
        for i in range(self.num_flights):
            if i not in self.done:
                for j in range(self.num_flights):
                    if j not in self.done and j != i:
                        # predicted used instead of position, so ownship can work in regard to future position and still
                        # avoid a future conflict
                        cur_dis[i][j] = self.flights[i].position.distance(self.flights[j].position)

                        distance_all[i][j] = self.flights[i].prediction.distance(self.flights[j].prediction)

                        # bearing
                        dx = self.flights[i].prediction.x - self.flights[j].prediction.x
                        dy = self.flights[i].prediction.y - self.flights[j].prediction.y
                        compass = math.atan2(dx, dy)
                        bearing_all[i][j] = (compass + u.circle) % u.circle

                        dx_all[i][j] = self.flights[i].position.x - self.flights[j].position.x
                        dy_all[i][j]  = self.flights[i].position.y - self.flights[j].position.y

        for i, f in enumerate(self.flights):
            if i not in self.done:
                observations = []

                closest_intruders = np.argsort(distance_all[i])[:NUMBER_INTRUDERS_STATE]

                # distance to closest #NUMBER_INTRUDERS_STATE
                observations += np.take(cur_dis[i], closest_intruders).tolist()

                # during training the number of flights may be lower than #NUMBER_INTRUDERS_STATE
                while len(observations) < NUMBER_INTRUDERS_STATE:
                    observations.append(0)
                
                observations += np.take(distance_all[i], closest_intruders).tolist()

                # during training the number of flights may be lower than #NUMBER_INTRUDERS_STATE
                while len(observations) < 2*NUMBER_INTRUDERS_STATE:
                    observations.append(0)

                # relative bearing #NUMBER_INTRUDERS_STATE
                observations += np.take(dx_all[i], closest_intruders).tolist()

                # during training the number of flights may be lower than #NUMBER_INTRUDERS_STATE
                while len(observations) < 3*NUMBER_INTRUDERS_STATE:
                    observations.append(0)

                observations += np.take(dy_all[i], closest_intruders).tolist()

                # during training the number of flights may be lower than #NUMBER_INTRUDERS_STATE
                while len(observations) < 4*NUMBER_INTRUDERS_STATE:
                    observations.append(0)
                
                
                

                
                # current speed
                observations.append(f.airspeed)

                # optimal speed
                observations.append(f.optimal_airspeed)

                # # distance to target
                # observations.append(f.position.distance(f.target))

                # bearing to target
                observations.append(m.sin(float(f.drift)))
                observations.append(m.cos(float(f.drift)))

                observations_all.append(observations)
        # RDC: here you should implement your observation function
        ##########################################################
        return observations_all
        ##########################################################

    def update_conflicts(self) -> None:
        """
        Updates the set of flights that are in conflict
        Note: flights that reached the target are not considered
        :return:
        """
        # reset set
        self.conflicts = set()

        for i in range(self.num_flights - 1):
            if i not in self.done:
                for j in range(i + 1, self.num_flights):
                    if j not in self.done:
                        distance = self.flights[i].position.distance(self.flights[j].position)
                        if distance < self.min_distance:
                            self.conflicts.update((i, j))

    def update_done(self) -> None:
        """
        Updates the set of flights that reached the target
        :return:
        """
        for i, f in enumerate(self.flights):
            if i not in self.done:
                distance = f.position.distance(f.target)
                if distance < self.tol:
                    self.done.add(i)

    def update_positions(self) -> None:
        """
        Updates the position of the agents
        Note: the position of agents that reached the target is not modified
        :return:
        """
        for i, f in enumerate(self.flights):
            if i not in self.done:
                # get current speed components
                dx, dy = f.components

                # get current position
                position = f.position

                # get new position and advance one time step
                f.position._set_coords(position.x + dx * self.dt, position.y + dy * self.dt)

    def step(self, action: List, isMARL, NUMBER_ACTORS_MARL) -> Tuple[List, List, bool, Dict]:
        """
        Performs a simulation step

        :param action: list of resolution actions assigned to each flight
        :return: observation, reward, done status and other information
        """
        # apply resolution actions
        self.resolution(action, isMARL)

        # update positions
        self.update_positions()

        # update done set
        self.update_done()

        # update conflict set
        self.update_conflicts()

        # compute reward
        rew = self.reward(isMARL)

        # compute observation
        obs = self.observation()

        while len(obs) < NUMBER_ACTORS_MARL:
            obs.append([0]*STATE_SIZE)

        # increase steps counter
        self.i += 1

        # check termination status
        # termination happens when
        # (1) all flights reached the target
        # (2) the maximum episode length is reached
        done = (self.i == self.max_episode_len) or (len(self.done) == self.num_flights)

        return obs, rew, done, {}

    def reset(self, number_flights_training, NUMBER_ACTORS_MARL) -> List:
        """
        Resets the environment and returns initial observation
        :return: initial observation
        """
        # create random airspace
        self.airspace = Airspace.random(self.min_area, self.max_area)

        # during training, the number of flights will increase from  1 to 10
        self.num_flights = number_flights_training

        # create random flights
        self.flights = []
        tol = self.distance_init_buffer * self.tol
        min_distance = self.distance_init_buffer * self.min_distance
        while len(self.flights) < self.num_flights:
            valid = True
            candidate = Flight.random(self.airspace, self.min_speed, self.max_speed, tol)

            # ensure that candidate is not in conflict
            for f in self.flights:
                if candidate.position.distance(f.position) < min_distance:
                    valid = False
                    break
            if valid:
                self.flights.append(candidate)

        # initialise steps counter
        self.i = 0

        # clean conflicts and done sets
        self.conflicts = set()
        self.done = set()

        # return initial observation
        obs = self.observation()
        while len(obs) < NUMBER_ACTORS_MARL:
            obs.append([0]*STATE_SIZE)

        return obs

    def render(self, mode=None) -> None:
        """
        Renders the environment
        :param mode: rendering mode
        :return:
        """
        if self.viewer is None:
            # initialise viewer
            screen_width, screen_height = 600, 600

            minx, miny, maxx, maxy = self.airspace.polygon.buffer(10 * u.nm).bounds
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds(minx, maxx, miny, maxy)

            # fill background
            background = rendering.make_polygon([(minx, miny),
                                                 (minx, maxy),
                                                 (maxx, maxy),
                                                 (maxx, miny)],
                                                filled=True)
            background.set_color(*BLACK)
            self.viewer.add_geom(background)

            # display airspace
            sector = rendering.make_polygon(self.airspace.polygon.boundary.coords, filled=False)
            sector.set_linewidth(1)
            sector.set_color(*WHITE)
            self.viewer.add_geom(sector)

        # add current positions
        for i, f in enumerate(self.flights):
            if i in self.done:
                continue

            if i in self.conflicts:
                color = RED
            else:
                color = BLUE

            circle = rendering.make_circle(radius=self.min_distance / 2.0,
                                           res=10,
                                           filled=False)
            circle.add_attr(rendering.Transform(translation=(f.position.x,
                                                             f.position.y)))
            circle.set_color(*BLUE)

            plan = LineString([f.position, f.target])
            self.viewer.draw_polyline(plan.coords, linewidth=1, color=color)
            prediction = LineString([f.position, f.prediction])
            self.viewer.draw_polyline(prediction.coords, linewidth=4, color=color)

            self.viewer.add_onetime(circle)

        self.viewer.render()        

    def close(self) -> None:
        """
        Closes the viewer
        :return:
        """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None