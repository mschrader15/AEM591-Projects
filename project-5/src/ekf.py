from typing import List 

import numpy as np
from scipy.stats import multivariate_normal

from .base import LTI
from .base import EKFStep
from .base import Radar


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
        x_k_k = x.T + (K_k @ y_k.T)

        # P posteriori
        P_k_k = P_priori - K_k @ S_k @ K_k.T
        self.P_posteriori = P_k_k.copy()

        # return the states that we want to plot
        return EKFStep(x_k_k=x_k_k, y_k=y_k, P_k_k=P_k_k, S_k=S_k, P_k_k_1=P_priori)

    def run(self, mode="") -> List[EKFStep]:

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
            
            if "silent" not in mode:
                print(i)
            
            i += 1

        return results

