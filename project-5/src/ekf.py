from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.stats import multivariate_normal

try:
    from base import LTI, Radar
except ModuleNotFoundError:
    # For the stupid jupyter format
    from .base import LTI, Radar


@dataclass
class EKFStep:
    x_k_k: np.ndarray
    y_k: np.ndarray
    P_k_k: np.ndarray
    P_k_k_1: np.ndarray
    S_k: np.ndarray


class EKF(LTI):
    def __init__(
        self,
        lti: LTI,
        R: np.ndarray,
        Q: np.ndarray,
        radars: List[
            Radar,
        ],
    ) -> None:

        # lets me initialize super with an already instantated class
        self.__dict__.update(lti.__dict__)

        self.radars = radars

        dim_x = self.x0.shape[0]
        dim_y = np.array(self.A_j).shape[1]

        self.x = self.x0

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

        # create a holder for the results
        self.results = []

    def step_car(self, i):
        try:
            return self.trajectory[i]
        except IndexError:
            return

    def update(self, x, measurement, *args, **kwargs) -> EKFStep:

        state = self.f_fast(
            *self.x,
        )  # dict(x=x[0], y=x[1], theta=x[2])

        y_act = measurement  # self.measure(**state, noise_matrix=self.R_func.rvs())

        F_j = self.F(*list(x), **kwargs)

        H_j = self.H(*state)

        # A Priori State Covariance
        P_priori = F_j @ self.P_posteriori @ F_j.T + self.L @ self.Q @ self.L.T

        # Innovation
        y_k = y_act - self.measure_fast(*state, noise_matrix=self.R_func.rvs())

        # Innovation Covariance
        S_k = H_j @ P_priori @ H_j.T + self.M @ self.R @ self.M.T

        # sub-optimal Kalman Gain
        K_k = P_priori @ H_j.T @ np.linalg.inv(S_k)

        # a posteriori mean estimate
        x_k_k = x + (K_k @ y_k.T)

        # P posteriori
        P_k_k = P_priori - K_k @ S_k @ K_k.T
        self.P_posteriori = P_k_k.copy()

        self.x = x_k_k

        # return the states that we want to plot
        return EKFStep(x_k_k=x_k_k, y_k=y_k, P_k_k=P_k_k, S_k=S_k, P_k_k_1=P_priori)

    def run(self, measurements: list = None, *args, **kwargs) -> List[EKFStep]:

        # could also use recursion to do this
        results = []
        i = 0
        while True:
            x_k = self.step_car(i)

            if x_k is None:
                break

            measurement = (
                self.measure_fast(*x_k, noise_matrix=self.R_func.rvs())
                if measurements is None
                else measurements[i]
            )

            results.append(self.update(x=x_k, measurement=measurement))

            i += 1

        self.results = results
        # return results


if __name__ == "__main__":

    """
    For testing purposes
    """

    from base import LTI, Radar
    from helpers import RAD_2_DEGREE, calculate_dubins, normalize_radians

    R_path = 5

    optimal_path = calculate_dubins()

    dt = 0.5
    radar_1 = Radar(x=-15, y=-10, v=9)
    radar_2 = Radar(x=-15, y=5, v=9)

    lti = LTI(
        s=1,
        s_var=0.05,
        dt=0.5,
        x0=optimal_path[0],
        dubins_path=optimal_path,
        q1=(-5, 20, -180),
        radar_1=radar_1,
        radar_2=radar_2,
    )

    lti.x_t_noise(
        x=[np.array(optimal_path[0])],
    )

    lti.trajectory = [
        np.array(
            (
                x[0],
                x[1],
                normalize_radians(x[2]),
            )
        )
        for x in lti.trajectory
    ]

    R = np.diag(
        [
            radar_1.v / (RAD_2_DEGREE ** 2),
            radar_2.v / (RAD_2_DEGREE ** 2),
            5 / (RAD_2_DEGREE ** 2),
        ]
    )
    Q = np.diag([0.05, 0.05, (1 / R_path) ** 2 * dt ** 2])

    ekf = EKF(lti, R=R, Q=Q, radars=(radar_1, radar_2))
    ekf.run()

    ekf.results[0].x_k_k
    # res[0].S_k
