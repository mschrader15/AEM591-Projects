import math
from typing import Callable, Tuple

import numpy as np

try:
    from base import BaseFilter
    from helpers import normalize_radians, RAD_2_DEGREE
except ModuleNotFoundError:
    # For the stupid jupyter format
    from .base import BaseFilter
    from .helpers import normalize_radians, RAD_2_DEGREE


class PF(BaseFilter):

    def __init__(self, num_p: int, x0: np.ndarray, dim_x: int, dim_y: int, R: np.ndarray, Q: np.ndarray, fx: Callable, hx: Callable, record_variables: list = None) -> None:
        
        super().__init__(x0, dim_x, dim_y, R, Q, fx, hx, record_variables=record_variables)

        # create the required objects:
        self._num_p = num_p
        self.state_p = np.zeros((self._num_p, self._dim_x))
        self.weights_p = np.zeros((self._num_p, self._dim_y))

        # self._state_p 
    
    def _update(self, *args, **kwargs) -> None:
        state_predict = np.zeros_like(self.state_p)
        for i, state_p_row in enumerate(self.state_p):
            state_predict[i] = self.fx(*state_p_row)
        
        for i, prediction in enumerate(self.state_p)
        


    def _predict(self, *args, **kwargs) -> None:

        for sp in state_p:


    def _resample_particles(self, particles: np.ndarray) -> None:
        cumsum_ = np.cumsum(self.weights_p)
        # binary search 
        indexes = np.searchsorted(np.random.rand(self._num_p), cumsum_)
        particles[:] = particles[indexes]

    def _reset_weights(self, ):
        self.weights_p = 1 / self._num_p * np.ones((self._num_p, 1))

    def _estimate(self, ):
        mean = np.average(self.state_p, weights=self.weights_p, axis=0)
        var = np.average(self.state_p - mean, weights=self.weights_p, axis=0)
        return mean, var

    def _roughen_particles(self, ):
        max_diff = abs(np.max(self.state_p) - np.min(self.state_p))
        self.state_p += np.sqrt(0.2 * max_diff / self._num_p) + np.random.randn(self._num_p)