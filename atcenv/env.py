"""
Environment module
"""
import gym
from typing import Dict, List
from atcenv.definitions import *
from gym.envs.classic_control import rendering
from shapely.geometry import LineString

# our own packages
import numpy as np

WHITE = [255, 255, 255]
GREEN = [0, 255, 0]
BLUE = [0, 0, 255]
BLACK = [0, 0, 0]
RED = [255, 0, 0]

# Position uncertainty vars
ENABLE_POSITION_UNCERTAINTY = False
PROB_POSITION_UNCERTAINTY = 0.2
MAG_POSITION_UNCERTAINTY = 500 # m

# Wind
ENABLE_WIND = False
MINIMUM_WIND_SPEED = 0 # m/s
MAXIMUM_WIND_SPEED = 30 # m/s

# Delay
ENABLE_DELAY = False
MAXIMUM_DELAY = 3 # s
PROB_DELAY = 0.1

NUMBER_INTRUDERS_STATE = 5
NUMBER_INTRUDERS_STATE = 2
MAX_DISTANCE = 250*u.nm
MAX_BEARING = math.pi

class Environment(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self,
                 num_flights: int = 1,
                 dt: float = 5.,
                 max_area: Optional[float] = 200. * 200.,
                 min_area: Optional[float] = 125. * 125.,
                 max_speed: Optional[float] = 500.,
                 min_speed: Optional[float] = 300,
                 max_episode_len: Optional[int] = 300,
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
        self.flights = FlightCollection([]) # list of flights
        self.conflicts = np.zeros((num_flights, num_flights))  # set of flights that are in conflict
        self.done = np.zeros((num_flights, 1))  # set of flights that reached the target
        self.i = None
        
        # Get the random wind direction and intensity for this episode
        self.wind_magnitude = random.randint(MINIMUM_WIND_SPEED, MAXIMUM_WIND_SPEED)
        self.wind_direction = random.randint(0, 359)

    def resolution(self, action: np.ndarray) -> None:
        """
        Applies the resolution actions
        If your policy can modify the speed, then remember to clip the speed of each flight
        In the range [min_speed, max_speed]
        :param action: list of resolution actions assigned to each flight
        :return:
        """
        
        # Vectorized implementation of the below code
        #new_track = self.flights.track + ((~self.done).astype(float) * np.expand_dims(action[:, 0], 1)) * MAX_BEARING / 8
        #self.flights.track = (new_track + u.circle) % u.circle

        self.flights.airspeed += ((~self.done).astype(float) * np.expand_dims(action[:, 0], 1)) * (self.max_speed - self.min_speed) / 3
        self.flights.airspeed = np.clip(self.flights.airspeed, self.min_speed, self.max_speed) # Limit airspeed to the limits
        
        # RDC: here you should implement your resolution actions
        ##########################################################
        return None
        ##########################################################

    def reward(self) -> np.ndarray:
        """
        Returns the reward assigned to each agent
        :return: reward assigned to each agent
        """
        weight_a    = -10 #-10
        weight_b    = 1/5.
        weight_c    = 0
        weight_d    = -0.001
        weight_e    = -5
        
        conflicts   = self.conflict_penalties() * weight_a
        drifts      = self.drift_penalties() * weight_b
        #severities  = self.conflict_severity() * weight_c 
        speed_dif   = self.speedDifference() * weight_d 
        target      = self.reachedTarget() * weight_e # can also try to just ouput negative rewards
        
        #tot_reward  = conflicts + drifts + severities + speed_dif + target  
        tot_reward = conflicts + drifts + speed_dif + target

        return np.squeeze(tot_reward)

    def reachedTarget(self):
        """
        Returns a list with aircraft that just reached the target
        :return: boolean list - 1 if aircraft have reached the target
        """

        # Vectorized code for the above portion
        return self.done

    def speedDifference(self):
        """
        Returns a list with the diferent betwee the current aircraft and its optimal speed
        :return: float of the speed difference
        """

        # Vectorized code for the above portion
        return np.abs(self.flights.airspeed - self.flights.optimal_airspeed)

    def conflict_penalties(self):
        """
        Returns a list with aircraft that are in conflict,
        can be used for multiplication as individual reward
        component
        :return: boolean list for conflicts
        """

        # Filter out done planes
        conflicts = self.conflicts * self.done.T.astype(float)

        # Get planes still in a conflict
        conflicts = np.expand_dims(np.sum(conflicts, axis=1), 1)

        return conflicts > 0

    def drift_penalties(self):
        """
        Returns a list with the drift angle for all aircraft,
        can be used for multiplication as individual reward
        component
        :return: float of the drift angle
        """
        return (.5 - np.abs(self.flights.drift)) * (~self.done).astype(float)

    def conflict_severity(self):
        distances = self.flights.distance_all

        # Make sure we don't divide by 0
        distances += .001

        # Remove planes that are done
        distances *= (~self.done).T.astype(float)
        
        # Remove planes not in conflict
        distances *= self.conflicts

        # Make distances of 0 now infinite so the severity goes to 0
        distances[distances == 0] = np.inf

        severity = -1. * (np.min(distances, axis=1) - self.min_distance)/self.min_distance

        severity[severity == np.inf] = 0
        
        return np.expand_dims(severity, 1)

    def observation(self) -> np.ndarray:
        """
        Returns the observation of each agent
        :return: observation of each agent
        """
        # observations (size = 2 * NUMBER_INTRUDERS_STATE + 5):
        # distance to closest #NUMBER_INTRUDERS_STATE intruders
        # relative bearing to closest #NUMBER_INTRUDERS_STATE intruders
        # current bearing
        # current speed
        # optimal airspeed
        # distance to target
        # bearing to target
        
        
        observations_all = []

        cur_dis = self.flights.distance_all
        distance_all = self.flights.predict_distance_all
        
        x_pos = np.expand_dims(self.flights.position[:, 0], 1)
        y_pos = np.expand_dims(self.flights.position[:, 1], 1)

        dx = x_pos.T - x_pos
        dy = y_pos.T - y_pos
        compass = np.arctan2(dx, dy)
        compass -= self.flights.track
        compass = (compass + u.circle) % u.circle
        
        bearing_all = compass.copy()
        np.putmask(bearing_all, compass > math.pi, -(u.circle - compass))
        np.putmask(bearing_all, compass < -math.pi, u.circle + compass)

        trackdif_all = self.flights.track - self.flights.track.T

        dx_all = np.sin(bearing_all) * cur_dis
        dy_all = np.cos(bearing_all) * cur_dis
        
        # The original loops keep certain values at the max, so we need to set those
        # as well
        mask = np.tile(self.done.T, (self.num_flights, 1))

        np.fill_diagonal(mask, 1)
        
        np.putmask(cur_dis, mask, MAX_DISTANCE)
        np.putmask(distance_all, mask, MAX_DISTANCE)
        np.putmask(bearing_all, mask, MAX_BEARING)
        np.putmask(dx_all, mask, MAX_DISTANCE)
        np.putmask(dy_all, mask, MAX_DISTANCE)
        
        closest_intruders = np.argsort(distance_all, axis=1)[:, :NUMBER_INTRUDERS_STATE]

        cur_dis_obs = np.take_along_axis(cur_dis, closest_intruders, axis=1)
        dist_all_obs = np.take_along_axis(distance_all, closest_intruders, axis=1)
        dx_all_obs = np.take_along_axis(dx_all, closest_intruders, axis=1)
        dy_all_obs = np.take_along_axis(dy_all, closest_intruders, axis=1)
        track_dif_obs = np.take_along_axis(trackdif_all, closest_intruders, axis=1)
        
        observations = np.concatenate([
                                        cur_dis_obs, 
                                        dist_all_obs, 
                                        dx_all_obs, 
                                        dy_all_obs, 
                                        track_dif_obs, 
                                        self.flights.airspeed, 
                                        self.flights.optimal_airspeed, 
                                        np.sin(self.flights.drift), 
                                        np.cos(self.flights.drift),
                                        self.done
                                        ], 
                                      axis=1)
        
        return observations

    def update_conflicts(self) -> None:
        """
        Updates the set of flights that are in conflict
        Note: flights that reached the target are not considered
        :return:
        """

        distances = self.flights.distance_all
        
        # Remove self collisions
        np.fill_diagonal(distances, np.inf)
        
        self.conflicts = distances < self.min_distance

        # Remove conflicts with done planes
        self.conflicts = np.logical_and(self.conflicts, ~self.done.T)
        
    def update_done(self) -> None:
        """
        Updates the set of flights that reached the target
        :return:
        """
        distance = self.flights.distance

        self.done = np.expand_dims(distance < self.tol, 1)

    def update_positions(self) -> None:
        """
        Updates the position of the agents
        Note: the position of agents that reached the target is not modified
        :return:
        """

        dx, dy = self.flights.components
        position = self.flights.position

        new_x = np.expand_dims(position[:, 0], 1) + (dx * self.dt * (~self.done).astype(float))
        new_y = np.expand_dims(position[:, 1], 1) + (dy * self.dt * (~self.done).astype(float))
        
        new_pos = np.concatenate([new_x, new_y], axis=1)
        self.flights.position = new_pos

        self.flights.reported_position = self.flights.position
        self.flights.prev_dx = dx
        self.flights.prev_dy = dy

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Performs a simulation step

        :param action: list of resolution actions assigned to each flight
        :return: observation, reward, done status and other information
        """

        # apply resolution actions
        self.resolution(action)

        # update positions
        self.update_positions()

        # update done set
        self.update_done()

        # update conflict set
        self.update_conflicts()

        # compute reward
        rew = self.reward()

        # compute observation
        obs = self.observation()

        # increase steps counter
        self.i += 1

        # store difference from optimal speed
        self.checkSpeedDif()

        # check termination status
        # termination happens when
        # (1) all flights reached the target
        # (2) the maximum episode length is reached
        done_t = np.expand_dims(np.array([self.i == self.max_episode_len for _ in range(self.num_flights)]), 1)
        done_e = self.done

        return obs, rew, done_t, done_e, {}

    def checkSpeedDif(self):
        self.average_speed_dif = 0
        speed_dif = np.abs(self.flights.airspeed - self.flights.optimal_airspeed)
        
        self.average_speed_dif = np.average(speed_dif)

    def reset(self) -> List:
        """
        Resets the environment and returns initial observation
        :return: initial observation
        """
        # create random airspace
        self.airspace = Airspace.random(self.min_area, self.max_area)

        # create random flights
        flights = []
        tol = self.distance_init_buffer * self.tol
        min_distance = self.distance_init_buffer * self.min_distance
        while len(flights) < self.num_flights:
            valid = True
            candidate = Flight.random(self.airspace, self.min_speed, self.max_speed, tol)

            # ensure that candidate is not in conflict
            for f in flights:
                if candidate.position.distance(f.position) < min_distance:
                    valid = False
                    break
            if valid:
                flights.append(candidate)
        # initialise steps counter
        self.i = 0

        # clean conflicts and done sets
        self.conflicts = np.zeros((self.num_flights, self.num_flights))
        self.done = np.zeros((self.num_flights, 1)).astype(bool)
        self.flights = FlightCollection(flights)

        # return initial observation
        return self.observation()

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
        for i in range(self.num_flights):
            if self.done[i]:
                continue

            if np.sum(self.conflicts[i, :]) > 0:
                color = RED
            else:
                color = BLUE

            circle = rendering.make_circle(radius=self.min_distance / 2.0,
                                           res=10,
                                           filled=False)
            circle.add_attr(rendering.Transform(translation=(self.flights.reported_position[i, 0],
                                                             self.flights.reported_position[i, 1])))
            circle.set_color(*BLUE)

            plan = LineString([self.flights.reported_position[i, :], self.flights.target[i, :]])
            self.viewer.draw_polyline(plan.coords, linewidth=1, color=color)
            prediction = LineString([self.flights.reported_position[i, :], self.flights.prediction[i, :]])
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
