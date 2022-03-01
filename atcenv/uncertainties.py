import random
from atcenv.definitions import Flight, Airspace
from shapely.geometry import Point, Polygon


def position_scramble(point : Point, probability : float, min_dist : float, max_dist : float):
    '''
    Input: a point. Output: a scrambled position. Must be within airspace.'''
    pass