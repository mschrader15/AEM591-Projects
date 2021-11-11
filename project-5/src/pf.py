import math
from typing import Callable, Tuple

import numpy as np
import scipy

try:
    from base import BaseFilter
    from helpers import normalize_radians, RAD_2_DEGREE
except ModuleNotFoundError:
    # For the stupid jupyter format
    from .base import BaseFilter
    from .helpers import normalize_radians, RAD_2_DEGREE


class PF(BaseFilter):

    def __init__(self, num_p: int, alpha: float, x0: np.ndarray, dim_x: int, dim_y: int, R: np.ndarray, Q: np.ndarray, fx: Callable, hx: Callable, record_variables: list = None) -> None:
        
        super().__init__(x0, dim_x, dim_y, R, Q, fx, hx, record_variables=["mean_x", "mean_var"])

        # alpha
        self._alpha = alpha

        # create the required objects:
        self._num_p = num_p

        # Intialize the particles and weights
        self.state_p = self._create_gaussian_particles(x0, std=self.Q.diagonal(), N=self._num_p)
        self.weights_p = np.ones(self._num_p) / self._num_p
        

        self._state_predict = np.zeros_like(self.state_p)
        self._state_measure = np.zeros_like(self.state_p)
        # for storing the variables
        self.mean_x = None
        self.mean_var = None

        # roughen divisor
        temp = np.arange(1, self._num_p + 1)
        self._roughen_d = np.c_[temp.copy(), temp.copy(), temp.copy()]

    def _update(self, measurement, *args, **kwargs) -> None:
        # computing the normalized weights
        for i, prediction in enumerate(self._state_predict):
            p_measure = self.hx(*prediction, )
            p_measure[2] = normalize_radians(p_measure[2])
            self._state_measure[i, :] = p_measure
        
        weights = scipy.stats.norm(self._state_measure, np.sqrt(self.R.diagonal())).pdf(measurement)
        self.weights_p *= np.prod(weights, 1)
        # self.weights_p += 1.e-300
        self.weights_p /= sum(self.weights_p)

        # resample the particles
        self._resample_particles()

        # calculate the mean
        self.mean_x, self.mean_var = self._estimate()

        # reset the weights
        self._reset_weights()

        # roughen the particles
        self._roughen_particles()
        
    def _predict(self, *args, **kwargs) -> None:
        # predict
        for i, state_p_row in enumerate(self.state_p):
            self._state_predict[i, :] = self.fx(*state_p_row, noise_matrix=self.Q_func.rvs())
            self._state_predict[i, 2] = normalize_radians(self._state_predict[i, 2])

    def _resample_particles(self, ) -> None:
        cumsum_ = np.cumsum(self.weights_p)
        # binary search
        idxs = np.searchsorted(cumsum_, np.random.rand(cumsum_.shape[0]))
        self.state_p = self._state_predict[idxs]

    def _reset_weights(self, ):
        self.weights_p = 1 / self._num_p * np.ones(self._num_p)

    def _estimate(self, ):
        mean = np.average(self.state_p, axis=0, weights=self.weights_p)
        mean[2] = self._average_angles(self.state_p[:, 2], self.weights_p)
        var = np.var(self.state_p, axis=0)  # weights=self.weights_p, axis=0)
        return mean, var

    def _roughen_particles(self, ):
        max_diff = abs(np.max(self.state_p, 0) - np.min(self.state_p, 0))
        self.state_p += np.sqrt(self._alpha * max_diff / self._roughen_d) * np.random.randn(self._num_p, self._dim_x)

    @staticmethod
    def _create_gaussian_particles(mean, std, N):
        particles = np.empty((N, 3))
        particles[:, 0] = mean[0] + (np.random.randn(N) * std[0])
        particles[:, 1] = mean[1] + (np.random.randn(N) * std[1])
        particles[:, 2] = mean[2] + (np.random.randn(N) * std[2])
        return particles

    @staticmethod
    def _average_angles(angles, weights):
        sin_sum = np.sum(np.sin(angles) * weights)
        cos_sum = np.sum(np.cos(angles) * weights)
        return normalize_radians(np.arctan2(sin_sum, cos_sum))


if __name__ == "__main__":

    """
    For testing purposes
    """

    from base import LTI, Radar
    from helpers import calculate_dubins
    
    R = 5

    optimal_path = calculate_dubins()

    dt = 0.5
    radar_1 = Radar(x=-15, y=-10, v=9)
    radar_2 = Radar(x=-15, y=5, v=9)

    lti = LTI(s=1, 
              s_var=0.05, 
              dt=dt,
              x0=optimal_path[0], 
              dubins_path=optimal_path, 
              q1=(-5, 20, -180), 
              radar_1=radar_1, 
              radar_2=radar_2)

    lti.x_t_noise(x=[np.array(optimal_path[0])], )

    lti.trajectory = \
        [np.array((x[0], x[1], normalize_radians(x[2]),)) for x in lti.trajectory] 

    pf = PF(
        num_p=1000,
        alpha=0.02,
        x0=lti.x0,
        dim_x=lti.A.shape[0],
        dim_y=lti.C.shape[0],
        R=np.diag([radar_1.v / (RAD_2_DEGREE ** 2), radar_2.v /
                  (RAD_2_DEGREE ** 2), 5 / (RAD_2_DEGREE ** 2)]),
        Q=np.diag([0.05, 0.05, (1 / R) ** 2 * dt ** 2]), 
        fx=lti.f_fast,
        hx=lti.measure_fast
        )

    pf.run(lti)

    pf.ss['K']
