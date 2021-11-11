import math
from typing import Tuple

import numpy as np
from scipy.linalg import cholesky

try:
    from base import BaseFilter
    from helpers import normalize_radians, RAD_2_DEGREE
except ModuleNotFoundError:
    # For the stupid jupyter format
    from .base import BaseFilter
    from .helpers import normalize_radians, RAD_2_DEGREE


class SigmaPoints:

    """
    SigmaPoints uses the Merwe Scaled algorithm
    """

    def __init__(self, dim: int, alpha: float, beta: float, kappa: float, ) -> None:

        self.dim = dim
        self.a = alpha
        self.b = beta
        self.kappa = kappa

        # only compute lambda once
        self._lambda = self.a ** 2 * (self.dim + self.kappa) - self.dim

        # compute the Wm and Wc matrices (this could be a tuning parameter)
        c = .5 / (self.dim + self._lambda)

        self.Wc = np.full(self.size, c)
        self.Wm = np.full(self.size, c)
        self.Wc[0] = self._lambda / \
            (self.dim + self._lambda) + (1 - self.a ** 2 + self.b)
        self.Wm[0] = self._lambda / (self.dim + self._lambda)

    def calc(self, mu: np.array, cov: np.array) -> np.matrix:

        u = cholesky((self._lambda + self.dim) * cov)

        sigma = np.zeros((2 * self.dim + 1, self.dim))
        sigma[0] = mu

        for i in range(self.dim):
            sigma[i + 1] = np.add(mu, u[i])
            sigma[self.dim + i + 1] = np.subtract(mu, u[i])

        return sigma

    @property
    def size(self, ) -> int:
        return self.dim * 2 + 1


class UKF(BaseFilter):
    """
    Base Filter contains most of the argument descriptions
    """

    def __init__(self, sigma_obj: SigmaPoints, *args, **kwargs) -> None:

        # initialize the parent
        super().__init__(
            *args, record_variables=['P_posteriori', 'P_priori', 'x_priori', 'x', 'K', 'z_res', 'z_measure', 'x_posteriori'], **kwargs)

        self.sp = sigma_obj

        self.sigmas_f = np.zeros((self.sp.size, self._dim_x))
        self.sigmas_h = np.zeros((self.sp.size, self._dim_y))

        self.x_mean = np.zeros((self._dim_x))
        self.x_priori = self.x_mean.copy()
        self.x_posteriori = self.x_mean.copy()

        self.K = np.zeros((self._dim_x, self._dim_y))
        self.z_res = np.zeros((self._dim_y))
        self.z_measure = np.array([[None] * self._dim_y]).T
        # self.S = np.zeros((self._dim_y, self._dim_y))
        # self.SI = np.zeros((self._dim_y, self._dim_y))

    def _predict(self, ) -> None:
        
        x = self.x.copy()

        sigma = self.sp.calc(mu=x, cov=self.P_posteriori)
        sigma_f = np.zeros_like(sigma)
        for i, s in enumerate(sigma):
            # sigma_f[i, :] = self.fx(*s, noise_matrix=self.Q_func.rvs()).T[0]
            sigma_f[i, :] = self.fx(*s, )
        
        # do the unscented transform on the sigma
        x, P_priori = self._unscented_transform(
            sigma_f,
            angle_ind=(2, ),
            noise_cov=self.Q
        )

        # update sigma points to reflect the new variance of the points
        self.sigmas_f = self.sp.calc(x, P_priori)

        # unsure as to why x and x_priori are set here
        self.x_mean = x.copy()
        self.x_priori = x.copy()

        self.P_priori = P_priori.copy()

    def _update(self, measurement: np.ndarray, *args, **kwargs) -> None:

        # pass prior sigmas through h(x) to get measurement sigmas
        # the shape of sigmas_h will vary if the shape of z varies, so
        # recreate each time
        sigmas_h = np.zeros_like(self.sigmas_f)
        for i, s in enumerate(self.sigmas_f):
            sigmas_h[i, :] = self.hx(*s, )  # noise_matrix=self.R_func.rvs())

        zp, S = self._unscented_transform(
            sigmas_h,
            angle_ind=(0, 1, 2),
            noise_cov=self.R
        )

        Pxz = self.cross_variance(
            self.x_priori, zp, self.sigmas_f, sigmas_h)

        self.K = np.dot(Pxz, np.linalg.inv(S))
        
        # subtract the measurement from the prediction and normalize the radians
        self.z_res = np.subtract(measurement, zp)
        self.z_res = normalize_radians(list(self.z_res))

        x = np.add(self.x_priori, np.dot(self.K, self.z_res))
        self.P_posteriori = self.P_priori - \
            np.dot(self.K, np.dot(S, self.K.T))

        # save measurement and posterior state
        self.z_measure = measurement.copy()
        self.x = x.copy()
        self.x_posteriori = x.copy()

    def _unscented_transform(self, sigma: np.ndarray, angle_ind: tuple, noise_cov: np.ndarray = None) -> Tuple[np.ndarray, ]:

        # x = np.dot(self.sp.Wm, sigma)
        x = self._angle_mean(sigma, angle_ind)
        # np.newaxis is a clever way to add a dimension
        y = sigma - x[np.newaxis, :]

        # normalize angles
        for i in angle_ind:
            y[:, i] = normalize_radians(list(y[:, i]))

        P = np.dot(y.T, np.dot(np.diag(self.sp.Wc), y))

        # add noise if it exists
        P = P + noise_cov if noise_cov is not None else P

        return x, P

    def cross_variance(self, x, z, sigmas_f, sigmas_h):

        Pxz = np.zeros((sigmas_f.shape[1], sigmas_h.shape[1]))

        for i in range(sigmas_f.shape[0]):
            Pxz += self.sp.Wc[i] * np.outer(
                np.subtract(sigmas_f[i], x),
                np.subtract(sigmas_h[i], z)
            )

        return Pxz

    def _angle_mean(self, sigmas: np.ndarray, angle_ind=(2, )) -> np.ndarray:
        """
        can't do a simple average because one of the states is an angle. 
        using the arctan of sines and cosines method
        This is annoyingly not general, as UKF doesn't know the position of the angles
        """
        x = np.zeros(sigmas.shape[1])

        for i in angle_ind:
            sin_sum = np.sum(np.dot(np.sin(sigmas[:, i]), self.sp.Wm))
            cos_sum = np.sum(np.dot(np.cos(sigmas[:, i]), self.sp.Wm))
            x[i] = math.atan2(sin_sum, cos_sum)

        for i in range(sigmas.shape[1]):
            if i not in angle_ind:
                x[i] = np.sum(np.dot(sigmas[:, i], self.sp.Wm))

        return x


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

    # Create the Sigma Point Object
    sp = SigmaPoints(dim=lti.A.shape[0], alpha=0.0001, beta=2, kappa=0)

    ukf = UKF(
        sigma_obj=sp,
        x0=lti.x0,
        dim_x=lti.A.shape[0],
        dim_y=lti.C.shape[0],
        R=np.diag([radar_1.v / (RAD_2_DEGREE ** 2), radar_2.v /
                  (RAD_2_DEGREE ** 2), 5 / (RAD_2_DEGREE ** 2)]),
        Q=np.diag([0.05, 0.05, (1 / R) ** 2 * dt ** 2]), 
        fx=lti.f_fast,
        hx=lti.measure_fast
        )

    ukf.P_posteriori = np.diag([.1, .1, .1])

    ukf.run(lti)

    ukf.ss['K']

    # res[0].S_k
