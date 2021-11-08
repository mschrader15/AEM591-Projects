from typing import Tuple

import numpy as np
from scipy.linalg import cholesky

from .base import BaseFilter


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
            *args, record_variables=['P_posteriori', 'P_priori', 'x'], **kwargs)

        self.sp = sigma_obj

        self.sigmas_f = np.zeros((self.sp.size, self._dim_x))
        self.sigmas_h = np.zeros((self.sp.size, self._dim_y))

        self.K = np.zeros((self._dim_x, self._dim_z))
        self.y = np.zeros((self._dim_y))
        self.z = np.array([[None] * self._dim_y]).T
        self.S = np.zeros((self._dim_y, self._dim_y))
        self.SI = np.zeros((self._dim_y, self._dim_y))

    def _update(self, measurement: np.ndarray) -> None:

        # pass prior sigmas through h(x) to get measurement sigmas
        # the shape of sigmas_h will vary if the shape of z varies, so
        # recreate each time
        sigmas_h = []
        for s in self.sigmas_f:
            sigmas_h.append(hx(s, **hx_args))


        # Need to rework this to my format
        self.sigmas_h = np.atleast_2d(sigmas_h)

        # mean and covariance of prediction passed through unscented transform
        zp, self.S = UT(self.sigmas_h, self.Wm, self.Wc, R, self.z_mean, self.residual_z)
        self.SI = self.inv(self.S)

        # compute cross variance of the state and the measurements
        Pxz = self.cross_variance(self.x, zp, self.sigmas_f, self.sigmas_h)


        self.K = dot(Pxz, self.SI)        # Kalman gain
        self.y = self.residual_z(z, zp)   # residual

        # update Gaussian state estimate (x, P)
        self.x = self.state_add(self.x, dot(self.K, self.y))
        self.P = self.P - dot(self.K, dot(self.S, self.K.T))

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()


    def _predict(self, ) -> None:
        x = self.x.copy()


        # do the unscented transform on the sigma
        x, P_posteriori = self._unscented_transform(
            self._transform_sigma(
                x
            )
        )

        # update sigma points to reflect the new variance of the points
        self.sigmas_f = self.sp.calc(x, P_posteriori)

        self.x = x.copy()
        self.P_posteriori = P_posteriori.copy()


    def _unscented_transform(self, sigma: np.ndarray) -> Tuple[np.ndarray, ]:

        x = np.dot(self.sp.Wm, sigma)
        # np.newaxis is a clever way to add a dimension
        y = sigma - x[np.newaxis, :]
        P = np.dot(y.T, np.dot(np.diag(self.sp.Wc), y))

        return x, P

    def _transform_sigma(self, x: np.ndarray) -> np.array:
        raw_sigma = self.sp.calc(mu=x, cov=self.P_posteriori)
        return np.array(map(self.fx, raw_sigma))


if __name__ == "__main__":

    ukf = UKF()
