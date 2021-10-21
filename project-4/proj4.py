import math
from dataclasses import dataclass
from typing import List, NamedTuple
from IPython.display import display

import sympy
import dubins
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
from scipy.interpolate import interp2d
from scipy.stats import multivariate_normal
from sympy.matrices import Matrix

sympy.init_printing(use_latex='mathjax')
pio.templates.default = "ggplot2"
pio.renderers.default = "plotly_mimetype"


@dataclass
class Radar:
    x: float
    y: float
    v: float


class LTI:

    def __init__(self, s: float, dt: float, x0: tuple, dubins_path: list, q1: tuple, s_var: float = None):

        # self.dubins_heading = interp2d(*[[d[i] for d in dubins_path] for i in range(3)])
        self.dubins_path = dubins_path
        self.q1 = q1
        self.s = s
        self.d_t = dt
        self.x0 = np.array(x0).T
        self.x_last = self.x0.copy()
        self.trajectory = []

        self.S_func = multivariate_normal(
            mean=self.s, cov=s_var) if s_var else None

        # re-import to avoid vairable naming issues. This is kind of nasty and not very PEP but :shrug:
        from sympy.abc import x, y, u, v, w, R, theta
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

    def measure(self, x: float, x1: float, y: float, y1: float, x2: float, y2: float, theta: float, noise_matrix: np.array = None) -> np.ndarray:
        measure = self._M_eval(self.C, x=x, x_1=x1, y=y, y_1=y1, x_2=x2, y_2=y2, theta=theta).T 
        noise = np.zeros_like(measure) if noise_matrix is None else noise_matrix
        return measure + noise

    def F(self, x: float, y: float, theta: float) -> np.ndarray:
        return self._M_eval(self.A_j, **{'x': x, 'y': y, 'theta': theta, 's': self.s, 'dt': self.d_t})

    def H(self, x: float, x1: float, y: float, y1: float, x2: float, y2: float, theta: float) -> np.ndarray:
        return self._M_eval(self.C_j, x=x, x_1=x1, y=y, y_1=y1, x_2=x2, y_2=y2, theta=theta)

    def f(self, x: float, y: float, theta: float, s: float = None) -> np.ndarray:
        return self._M_eval(self.A, **{'x': x, 'y': y, 'theta': theta, 's': s or self.s, 'dt': self.d_t})

    # def h(self, x: float, y: float, theta: float) -> np.ndarray:
    #     return self._M_eval(self.C, x=x, y=y, theta=theta)

    def x_t(self, x: float, y: float, theta: float) -> np.ndarray:
        """ Legacy Function Name"""
        return self.f(x, y, theta)

    @staticmethod
    def _find_nearest_dudin(dubin_path: list, x: float, y: float) -> int:
        distance = [((_p[0] - x) ** 2 + (_p[1] - y) ** 2) ** (1/2)
                    for _p in dubin_path]
        return np.argmin(distance)

    def x_t_noise(self, x: float, i=0, ) -> np.ndarray:
        while True:
            idx = self._find_nearest_dudin(
                self.dubins_path, x[-1][0][0], x[-1][0][1])
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
            # recursion excursion
            return self.x_t_noise(x, i+1)

        self.trajectory = x


@dataclass
class EKFStep:
    x_k_k: np.ndarray
    y_k: np.ndarray
    P_k_k: np.ndarray
    S_k: np.ndarray



class EKF(LTI):

    def __init__(self, lti: LTI, R: np.ndarray, Q: np.ndarray, radars: List[Radar, ]) -> None:
        
        # lets me initialize super with an already instantated class
        self.__dict__.update(lti.__dict__)

        self.radars = radars

        dim_x = self.x0.shape[0]
        dim_y = np.array(self.A_j).shape[1]

        self.x = np.zeros_like(self.x0)

        self.P_posteriori = np.eye(dim_x)
        self.P_priori = self.P_posteriori.copy()

        # initialize empty matrices
        self.K = np.zeros((dim_x, dim_y))
        self.I = np.eye(dim_x)

        # uncertainty
        self.R = R
        self.Q = Q

        # create a noise function
        self.R_func = multivariate_normal(mean=np.zeros(self.R.shape[0]), cov=self.R)

        # Gain Matrices
        self.L = np.eye(self.Q.shape[0])
        self.M = np.eye(self.R.shape[0])

    def step_car(self, i):
        try:
            return self.trajectory[i]
        except IndexError:
            return

    def update(self, x, *args, **kwargs) -> EKFStep:

        # x = self.step_car(iter_)
        state = dict(x=x[0][0], y=x[0][1], x1=self.radars[0].x, x2=self.radars[1].x,
                     y1=self.radars[0].y, y2=self.radars[1].y, theta=x[0][2])

        y_act = self.measure(**state, noise_matrix=self.R_func.rvs())

        F_j = self.F(*list(x[0]), **kwargs)

        H_j = self.H(**state)

        # A Priori State Covariance
        P_priori = F_j @ self.P_posteriori @ F_j.T + self.L @ self.Q @ self.L.T

        # Innovation
        y_k = y_act - self.measure(**state)

        # Innovation Covariance
        S_k = H_j @ P_priori @ H_j.T + self.M @ self.R @ self.M.T

        # sub-optimal Kalman Gain
        K_k = P_priori @ H_j.T @ np.linalg.inv(S_k)

        # a posteriori mean estimate
        x_k_k = x + (K_k @ y_k.T)

        # P posteriori
        P_k_k = P_priori - K_k @ S_k @ K_k.T
        self.P_posteriori = P_k_k.copy()

        # return the states that we want to plot
        return EKFStep(x_k_k=x_k_k, y_k=y_k, P_k_k=P_k_k, S_k=S_k)

    def run(self, ) -> List[EKFStep]:

        # could also use recursion to do this
        results = []
        i = 0
        while True:
            x_k = self.step_car(i)
            if x_k is None:
                break

            results.append(
                self.update(x=x_k, )
            )
            i += 1
            print(i)

        return results


if __name__ == "__main__":

    rad_to_deg = 180 / math.pi
    R = 5  # given
    dt = 0.01

    q0 = (0, -15, -90 / rad_to_deg)
    q1 = (-5, 20, -180 / rad_to_deg)
    turning_radius = R
    step_size = dt

    path = dubins.shortest_path(q0, q1, R)
    configurations, x = path.sample_many(step_size)

    radar_1 = Radar(x=-15, y=-10, v=9)
    radar_2 = Radar(x=-15, y=5, v=9)

    lti = LTI(s=1, s_var=0.05, dt=0.5,
              x0=configurations[0], dubins_path=configurations, q1=q1)
    lti.x_t_noise(x=[np.array([configurations[0]])], )

    e = EKF(lti, R=np.diag([9 / rad_to_deg, 9 / rad_to_deg, 5 / rad_to_deg]),
            Q=np.diag([0.05, 0.05, (1 / R) ** 2 * dt ** 2]), radars=(radar_1, radar_2))

    res = e.run()

    res[0].S_k
