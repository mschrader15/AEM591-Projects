import math
from typing import List, Union

import numpy as np

RAD_2_DEGREE = 180 / math.pi


def calculate_dubins(x0: tuple = (0, -15, -90), x1: tuple = (-5, 20, -180), r: int = 5, step_size: float = 0.01) -> list:
    import dubins

    x0[-1] = x0[-1] / RAD_2_DEGREE
    x1[-1] = x1[-1] / RAD_2_DEGREE
    path = dubins.shortest_path(x0, x1, r)
    optimal_path, _ = path.sample_many(step_size)
    return optimal_path


def normalize_radians(x: Union[List, float]) -> Union[List, float]:

    if isinstance(x, list):
        return list(map(normalize_radians, x))
    else:
        x = x % (2 * np.pi)
        if x > np.pi:
            x -= 2 * np.pi
        return x
