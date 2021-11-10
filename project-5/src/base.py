from typing import Callable
from dataclasses import dataclass

import sympy
from sympy.matrices import Matrix
import numpy as np
from scipy.stats import multivariate_normal


@dataclass
class Radar:
    x: float
    y: float
    v: float


class LTI:

    def __init__(self, s: float, dt: float, x0: tuple, dubins_path: list, q1: tuple, radar_1: Radar, radar_2: Radar, s_var: float = None, ):

        self.dubins_path = dubins_path
        self.q1 = q1
        self.s = s
        self.d_t = dt
        self.x0 = np.array(x0).T
        self.x_last = self.x0.copy()
        self.trajectory = []

        # save the radars
        self.radar_1 = radar_1
        self.radar_2 = radar_2

        self.S_func = multivariate_normal(
            mean=self.s, cov=s_var) if s_var else None

        # re-import to avoid vairable naming issues. This is kind of nasty and not very PEP but :shrug:
        from sympy.abc import x, y, u, theta
        from sympy import symbols, Matrix

        d_t, s = symbols('dt, s')
        y_1, x_1, y_2, x_2 = symbols('y_1, x_1, y_2, x_2')

        self.A = Matrix([
            [x + d_t * s * sympy.cos(theta)],
            [y + d_t * s * sympy.sin(theta)],
            [theta]
        ])

        self.B = Matrix([
            [0],
            [0],
            [u * d_t]
        ])

        self.A_j = self.A.jacobian(Matrix([x, y, theta]))

        self.C = Matrix([
            [sympy.atan2((y - y_1), (x - x_1))],
            [sympy.atan2((y - y_2), (x - x_2))],
            [theta]
        ])

        self.C_j = self.C.jacobian(Matrix([x, y, theta]))

    def _M_eval(self, M: Matrix, **kwargs) -> np.ndarray:
        return np.array(
            M.evalf(subs=kwargs)
        ).astype(float)

    def measure(self, x: float, y: float, theta: float, noise_matrix: np.array = None) -> np.ndarray:
        measure = self._M_eval(self.C, x=x, x_1=self.radar_1.x, y=y, y_1=self.radar_1.y,
                               x_2=self.radar_2.x, y_2=self.radar_2.y, theta=theta).T
        noise = np.zeros_like(
            measure) if noise_matrix is None else noise_matrix
        return measure + noise

    def F(self, x: float, y: float, theta: float) -> np.ndarray:
        return self._M_eval(self.A_j, **{'x': x, 'y': y, 'theta': theta, 's': self.s, 'dt': self.d_t})

    def H(self, x: float, y: float, theta: float) -> np.ndarray:
        return self._M_eval(self.C_j, x=x, x_1=self.radar_1.x, y=y, y_1=self.radar_1.y, x_2=self.radar_2.x, y_2=self.radar_2.y, theta=theta)

    def f(self, x: float, y: float, theta: float, s: float = None) -> np.ndarray:
        return self._M_eval(self.A, **{'x': x, 'y': y, 'theta': theta, 's': s or self.s, 'dt': self.d_t})

    def x_t(self, x: float, y: float, theta: float) -> np.ndarray:
        """ Legacy Function Name"""
        return self.f(x, y, theta)

    @staticmethod
    def _find_nearest_dubin(dubin_path: list, x: float, y: float) -> int:
        distance = [((_p[0] - x) ** 2 + (_p[1] - y) ** 2) ** (1/2)
                    for _p in dubin_path]
        return np.argmin(distance)

    def x_t_noise(self, x: list, ) -> np.ndarray:

        while True:

            idx = self._find_nearest_dubin(
                self.dubins_path,
                x[-1][0][0],
                x[-1][0][1]
            )

            du = self.dubins_path[idx][2]

            # why 2? Because that seems to end closet to the actual end
            if idx >= (len(self.dubins_path) - 2):
                break

            x.append(
                self.f(
                    x=x[-1][0][0],
                    y=x[-1][0][1],
                    theta=du,
                    s=self.S_func.rvs()
                ).T
            )

        # this is annoying but don't care to trace down the dimension issue
        self.trajectory = [_x[0] for _x in x]


class RecordableFilter:
    """
    This class assists in recording the states of filters over time
    """

    def __init__(self, record_variables) -> None:

        self.ss = {var: [] for var in record_variables}

    def _record(self, ) -> None:
        for key in self.ss.keys():
            self.ss[key].append(self.__dict__[key])

    def _update(self, *args, **kwargs) -> None:
        """
        Just a placeholder
        """
        pass

    def _predict(self, *args, **kwargs) -> None:
        """
        Just a placeholder
        """
        pass

    def predict(self, *args, **kwargs) -> None:
        self._predict(*args, **kwargs)

    def update(self, *args, **kwargs) -> None:
        self._update(*args, **kwargs)
        self._record()


class BaseFilter(RecordableFilter):

    def __init__(self, x0: np.ndarray, dim_x: int, dim_y: int, R: np.ndarray, Q: np.ndarray, fx: Callable, hx: Callable, record_variables: list = None,) -> None:

        super().__init__(record_variables)

        # --
        self.x = x0
        self.x_prior = None  # for logging purposes

        # --
        self.fx = fx
        self.hx = hx

        # ---
        self._dim_x = dim_x
        self._dim_y = dim_y

        # ---
        self.R = R.copy()
        self.Q = Q.copy()
        self.R_func = multivariate_normal(
            mean=np.zeros(self.R.shape[0]), cov=self.R)
        # ----
        self.I = np.eye(dim_x)
        self.P_posteriori = self.I.copy()
        self.P_priori = self.I.copy()
        self.K = np.zeros((dim_x, dim_y))

    def run(self, lti: LTI, ) -> None:

        for i, x_pos in enumerate(lti.trajectory):

            if i < 1:
                self.x = x_pos  # initialize to the initial position

            # predict the state
            self.predict()

            # take a measurement given the position
            measurement = lti.measure(*x_pos, self.R_func.rvs())[0]

            # update the prediction given the measurement
            self.update(measurement, hx_args={"noise_matrix": self.R_func.rvs()})
