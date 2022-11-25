"""
Definitions module
"""
from shapely.geometry import Point, Polygon
from dataclasses import dataclass, field
import atcenv.units as u
import math
import random
import numpy as np
from typing import Optional, Tuple, List

@dataclass
class Airspace:
    """
    Airspace class
    """
    polygon: Polygon

    @classmethod
    def random(cls, min_area: float, max_area: float):
        """
        Creates a random airspace sector with min_area < area <= max_area

        :param max_area: maximum area of the sector (in nm^2)
        :param min_area: minimum area of the sector (in nm^2)
        :return: random airspace
        """
        R = math.sqrt(max_area / math.pi)

        def random_point_in_circle(radius: float) -> Point:
            alpha = 2 * math.pi * random.uniform(0., 1.)
            r = radius * math.sqrt(random.uniform(0., 1.))
            x = r * math.cos(alpha)
            y = r * math.sin(alpha)
            return Point(x, y)

        p = [random_point_in_circle(R) for _ in range(3)]
        polygon = Polygon(p).convex_hull

        while polygon.area < min_area:
            p.append(random_point_in_circle(R))
            polygon = Polygon(p).convex_hull

        return cls(polygon=polygon)


@dataclass
class Flight:
    """
    Flight class
    """
    position: Point
    target: Point
    optimal_airspeed: float

    airspeed: float = field(init=False)
    track: float = field(init=False)
    
    prev_dx: float = field(init=False)
    prev_dy: float = field(init=False)
    
    reported_position: Point = None

    def __post_init__(self) -> None:
        """
        Initialises the track and the airspeed
        :return:
        """
        self.track = self.bearing
        self.airspeed = self.optimal_airspeed
        
        # Initialise previous speeds
        self.prev_dx = self.airspeed * math.sin(self.track)
        self.prev_dy = self.airspeed * math.cos(self.track)
        
        # The reported position, not the actual one
        self.reported_position = self.position
        # The action delay still left before 
        self.action_delay = -999
        self.delayed_action = None

    @property
    def bearing(self) -> float:
        """
        Bearing from current position to target
        :return:
        """
        if self.reported_position is not None:
            dx = self.target.x - self.reported_position.x
            dy = self.target.y - self.reported_position.y
        else:
            dx = self.target.x - self.position.x
            dy = self.target.y - self.position.y
        compass = math.atan2(dx, dy)
        return (compass + u.circle) % u.circle

    @property
    def prediction(self, dt: Optional[float] = 20) -> Point:
        """
        Predicts the future position after dt seconds, maintaining the current speed and track
        :param dt: prediction look-ahead time (in seconds)
        :return:
        """
        dx, dy = self.components
        return Point([self.reported_position.x + dx * dt, self.reported_position.y + dy * dt])

    @property
    def components(self) -> Tuple:
        """
        X and Y Speed components (in kt)
        :return: speed components
        """
        dx = self.airspeed * math.sin(self.track)
        dy = self.airspeed * math.cos(self.track)
        return dx, dy

    @property
    def distance(self) -> float:
        """
        Current distance to the target (in meters)
        :return: distance to the target
        """
        return self.reported_position.distance(self.target)

    @property
    def drift(self) -> float:
        """
        Drift angle (difference between track and bearing) to the target
        :return:
        """
        drift = self.bearing - self.track

        if drift > math.pi:
            return -(u.circle - drift)
        elif drift < -math.pi:
            return (u.circle + drift)
        else:
            return drift

    @classmethod
    def random(cls, airspace: Airspace, min_speed: float, max_speed: float, tol: float = 0.):
        """
        Creates a random flight

        :param airspace: airspace where the flight is located
        :param max_speed: maximum speed of the flights (in kt)
        :param min_speed: minimum speed of the flights (in kt)
        :param tol: tolerance to consider that the target has been reached (in meters)
        :return: random flight
        """
        def random_point_in_polygon(polygon: Polygon) -> Point:
            minx, miny, maxx, maxy = polygon.bounds
            while True:
                point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
                if polygon.contains(point):
                    return point

        # random position
        position = random_point_in_polygon(airspace.polygon)

        # random target
        boundary = airspace.polygon.boundary
        while True:
            d = random.uniform(0, airspace.polygon.boundary.length)
            target = boundary.interpolate(d)
            if target.distance(position) > tol:
                break

        # random speed
        airspeed = random.uniform(min_speed, max_speed)

        return cls(position, target, airspeed)

class FlightCollection:
    '''
    A collection of flights for vectorization
    '''
    def __init__(self, flights: List[Flight]):
        self.position = np.stack([np.array([f.position.x, f.position.y], dtype=np.float64) for f in flights])
        self.target = np.stack([np.array([f.target.x, f.target.y], dtype=np.float64) for f in flights])
        self.optimal_airspeed = np.array([f.optimal_airspeed for f in flights], dtype=np.float64)
        
        self.reported_position = self.position
        
        self.track = self.bearing
        self.airspeed = self.optimal_airspeed
        
        # Initialise previous speeds
        self.prev_dx = self.airspeed * np.sin(self.track)
        self.prev_dy = self.airspeed * np.cos(self.track)
        
        # The action delay still left before 
        self.action_delay = -999
        self.delayed_action = None

    @property
    def bearing(self) -> np.ndarray:
        """
        Bearing from current position to target
        :return:
        """
        if self.reported_position is not None:
            dx = self.target[:, 0] - self.reported_position[:, 0]
            dy = self.target[:, 1] - self.reported_position[:, 1]
        else:
            dx = self.target[:, 0] - self.position[:, 0]
            dy = self.target[:, 1] - self.position[:, 1]

        compass = np.arctan2(dx, dy)
        return (compass + u.circle) % u.circle
    
    @property
    def prediction(self, dt: Optional[float] = 20) -> np.ndarray:
        """
        Predicts the future position after dt seconds, maintaining the current speed and track
        :param dt: prediction look-ahead time (in seconds)
        :return:
        """
        dx, dy = self.components
        
        x_pred = self.reported_position[:, 0] + dx * dt
        y_pred = self.reported_position[:, 1] + dy * dt

        return np.stack([x_pred, y_pred], axis=1)

    @property
    def components(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        X and Y Speed components (in kt)
        :return: speed components
        """
        dx = self.airspeed * np.sin(self.track)
        dy = self.airspeed * np.cos(self.track)

        return dx, dy

    @property
    def distance(self) -> np.ndarray:
        """
        Current distance to the target (in meters)
        :return: distance to the target
        """
        return np.sqrt(np.sum(np.square(self.reported_position - self.target), axis=1))

    @property
    def drift(self) -> np.ndarray:
        """
        Drift angle (difference between track and bearing) to the target
        :return:
        """
        drift = self.bearing - self.track
        
        return_array = np.copy(drift)
        
        if len(return_array[drift > math.pi]) > 0:
            return_array[drift > math.pi] = -(u.circle - drift)
        if len(return_array[drift < -math.pi]) > 0:
            return_array[drift < -math.pi] = u.circle + drift
        
        return return_array
