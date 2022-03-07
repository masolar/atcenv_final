import random
import numpy as np
from atcenv.definitions import Flight, Airspace
from shapely.geometry import Point, Polygon


def position_scramble(ac_point, probability, min_dist, max_dist, alt = 0):
    '''
    Input: a point, a probability that the point. Output: a scrambled position. Must be within airspace.'''
    if min_dist > max_dist:
        raise Exception('Minimum distance cannot be greater than maximum distance.')
    
    if probability < 0 or probability > 1:
        raise Exception('Probability must be between 0 and 1.')
    
    # First of all, do we even do this. Do the probability
    probability_roll = random.random()
    if probability_roll >= probability:
        # Roll failed, return same position
        return ac_point
    
    else:
        # We scrable the position in a random direction
        random_dir = random.random() * 360

        # By a random magnitude
        random_mag = min_dist + random.random() * (max_dist - min_dist)
        
        # X direction
        new_x = ac_point.x + random_mag * np.cos(np.deg2rad(random_dir))
        new_y = ac_point.y + random_mag * np.sin(np.deg2rad(random_dir))
        
        return Point(new_x, new_y)

def apply_wind(flight, intensity, track):
    '''Apply the current wind to aircraft.'''
    wind_dx = intensity * np.sin(np.deg2rad(track))
    wind_dy = intensity * np.cos(np.deg2rad(track))
    
    # Get the components of the aircraft
    dx, dy = flight.components
    
    return dx + wind_dx, dy + wind_dy