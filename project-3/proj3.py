from typing import *
from control.statesp import ss

import numpy as np
from scipy.integrate import solve_ivp
import scipy.linalg as la
from control import StateSpace
from control import lqr
from control.matlab import lsim
# from scipy.signal import StateSpace, lsim
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.templates.default = "ggplot2"
pio.renderers.default = "jupyterlab"


class Plane:
    MAX_STEP = 0.1

    def __init__(self, ) -> None:
        self.A = np.array([
            [-0.038, 18.984, 0, -32.174, 0, 0, 0, 0],
            [-0.001, -0.632, 1, 0, 0, 0, 0, 0],
            [0, -0.759, -0.518, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -0.0829, 0, -1, 0.0487],
            [0, 0, 0, 0, -4.546, -1.699, 0.1717, 0],
            [0, 0, 0, 0, 3.382, -0.0654, -0.0893, 0],
            [0, 0, 0, 0, 0, 1, 0, 0]
        ])

        self.B = np.array([
            [0, 10.1, 0, 0],
            [-0.0086, 0, 0, 0],
            [-0.011, 0.025, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0.0116],
            [0, 0, 27.276, 0.5758],
            [0, 0, 0.3952, -1.362],
            [0, 0, 0, 0],
        ])

        self.Q = np.eye(self.A.shape[0])

        self.R = np.eye(self.B.shape[1]) * 0.01

        self.u_bar = 250  # ft/s

        self.x0 = np.array(
            [250, 0, 0, 0, 0, 0, 0, 0]
        )

    def optimal_f(self, time_vector: np.ndarray, target_states: List[np.ndarray], Q: np.ndarray, R: np.ndarray) -> object:
        
        A = np.r_[np.c_[self.A, np.zeros(self.A.shape[0])], [[1, 0, 0, 0, 0, 0, 0, 0, 0]]]
        B = np.r_[self.B, [[0] * self.B.shape[1]]]
#         C = np.eye(A.shape[0])
#         K, _, _ = lqr(A, B, Q, self.R)
        K = self.design_lrq(A, B, Q, R)

        if target_states[0].shape[0] != B.shape[0]:
            target_states = [np.append(_t, [0]) for _t in target_states]
            
        # creating the target function
        t_div = (time_vector[-1] - time_vector[0]) / len(target_states)
        t_f = lambda t: target_states[int(t // t_div)]
        
        Aaug = A-B@K
        X0 = np.append(self.x0, [0])

        def f(t, x_t, ):
            wr = t_f(t)
            return np.array(((Aaug @ (x_t - wr).T) - (wr - X0)))[0]
        
        return f

    def augA(self, ) -> np.ndarray:
        self.A
    
    @property
    def ss(self, ) -> StateSpace:
        return StateSpace(self.A, self.B, np.eye(self.A.shape[0]), np.zeros(self.B.shape))

    def simulate_uncontrolled(self, t_vect: np.ndarray, u_vect: np.ndarray, ) -> Tuple[np.ndarray]:
        self.T = t_vect
        self.U = u_vect
        return solve_ivp(self._f(), y0=self.x0, t_span=(t_vect[0], t_vect[-1]), max_step=Plane.MAX_STEP, args=(None, ))
#         return lsim(self.ss, u_vect, t_vect, self.x0)

    def simulate_controlled(self, time_vector: np.ndarray, *args, **kwargs) -> Tuple[np.ndarray]:
        # this is a bad practive but getting lazy
        augx0 = np.append(self.x0, [0])
        return solve_ivp(self.optimal_f(time_vector, *args, **kwargs), y0=augx0, t_span=(time_vector[0], time_vector[-1]), max_step=Plane.MAX_STEP)

    def _f(self, ):
        def f(t, x_t, wr):
            U = np.array(-1 * self.K @ (x_t - wr))[0] if wr is not None else np.array(self.step_function(t))
            return self.A @ x_t + (self.B @ U.T)
        return f

    def step_function(self, t: float):
        return self.U[int(t // ((self.T[-1] - self.T[0]) / len(self.U)))]

    @staticmethod
    def design_lrq(A, B, Q, R) -> None:
        X = np.matrix(
            la.solve_continuous_are(
                A, B, Q, R, e=None, balanced=True)
        )

        # compute the LQR gain
        K = np.matrix(la.inv(R) @ (B.T@ X))

        eigVals, eigVecs = la.eig(A-B*K)

        return K

    def control_lqr(self, ) -> np.ndarray:
        # This function is used as a check of my own LQR implementation
        from control import lqr

        return lqr(self.ss, self.Q, self.R)

    @staticmethod
    def plot(t: np.ndarray, y: np.ndarray, u: np.ndarray, flat: bool = True, target: tuple = None) -> None:
        fig = make_subplots(rows=5,
                            cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.05,
                            subplot_titles=("Airspeed",
                                            "Angular Velocity",
                                            "Angles",
                                            "Control Surface Deflection",
                                            "Thrust",
                                            )
                            )

        for j, (data, yaxis_title, inner_settings) in enumerate(
            [[y, "ft/s", [["u", 0]]],
             [y, "rad/s", [["q", 2], ["r", 6], ["p", 5]]],
             [y, "rad", [["phi", 7], ["theta", 3], ["beta", 4], ["alpha", 1]]],
             [u, "rad", [["elevator", 0], ["airlerons", 2], ["rudder", 3]]],
             [u, "lbs", [["thrust", 1]]],
             ]
        ):
            for name, ind in inner_settings:
                fig.add_trace(go.Scatter(
                    x=t,
                    #                     y=[val[ind] + self.u_bar if j < 1 else val[ind] for val in data],
                    y=data[ind] if flat else [val[ind] for val in data],
                    name=name
                ),
                    row=j + 1, col=1
                )
                
                if target and j < 3:
                    fig.add_trace(
                        go.Scatter(
                        x=target[0],
                        y=[val[ind] for val in target[1]],
                        line_color="black",
                        line_dash="dash",
                        name="Target",
                        showlegend=j < 1 
                    ),
                    row=j + 1, 
                    col=1,
                )

            update_dict = {
                f"yaxis{j+1}" if j > 0 else "yaxis": dict(title=yaxis_title)
            }
            fig.update_layout(update_dict)
        fig.update_layout(
            height=800
        )

        fig.show()



if __name__ == "__main__":

    p = Plane()

    # p.build_aug_matrixes(np.array([260, 0, 0, 0, 0, 0, 0, 0, 0]))

    # K = p.design_lrq()
    res = p.simulate_controlled(time_vector=np.linspace(0, 20, 100), target_states=[np.array([250, 0, 0, 0, 0, 0, 0, 0])] * 20 +  [np.array([260, 0, 0, 0, 0, 0, 0, 0])] * 80, 
    Q=np.diag([10] * 9), R=np.diag([20, 20, 20, 20]))
    # print(res)

    # u = np.array([[0, 0, 0, 0]] * 10 + [[0, 10, 0, 0]] * 90)
    # t, y, s = p.simulate_uncontrolled(np.linspace(0, 20, 100), u)
