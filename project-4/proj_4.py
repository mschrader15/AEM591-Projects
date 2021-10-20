import math
import sympy
import dubins
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
from sympy import symbols, Matrix
from scipy.interpolate import interp2d

from IPython.display import display

sympy.init_printing(use_latex='mathjax')
pio.templates.default = "ggplot2"
pio.renderers.default = "jupyterlab"


rad_to_deg = 180 / math.pi

R = 5  # given
dt = 0.5
q0 = (0, -15, -90 * rad_to_deg)
q1 = (-5, 20, -180 * rad_to_deg)
turning_radius = R
step_size = dt
path = dubins.shortest_path(q0, q1, turning_radius)
configurations, x = path.sample_many(step_size)



class LTI:
    
    def __init__(self, s: float, dt: float, x0: tuple, dubins_path: list):
        
        self.dubins_heading = interp2d(*[[d[i] for d in dubins_path] for i in range(3)])
        
        self.s = s
        self.d_t = dt
        self.x0 = np.array(x0).T
        self.x_last = self.x0.copy()

        # re-import to avoid vairable naming issues 
        from sympy.abc import x, y, u, v, w, R, theta
    
        d_t, s = symbols('dt, s')
        y_1, x_1, y_2, x_2 = symbols('y_1, x_1, y_2, x_2')
        self.A = Matrix([
            [x + d_t * s * sympy.cos(theta)],
            [y + d_t * s * sympy.sin(theta)],
            [theta + d_t * u]
        ])
        self.A_j = self.A.jacobian(Matrix([x, y, theta]))
        
        self.C = Matrix([
            [sympy.atan2((y - y_1), (x - x_1))],
            [sympy.atan2((y - y_2), (x - x_2))],
            [theta]
        ])
        self.C_j = self.C.jacobian(Matrix([x, y, theta]))
    
    def F(self, x, y, theta) -> np.ndarray:
        return np.array(self.A_j.evalf(subs={
                'x': x,
                'y': y,
                'theta': theta,
                's': self.s,
                'dt': self.d_t
        })).astype(float)
    
    def H(self, x, y, theta) -> np.ndarray:
        return np.array(self.C_j.evalf(subs={
                'x': x,
                'y': y,
                'theta': theta
        })).astype(float)
    
    def x_t(self, x, y, theta) -> np.ndarray:
        x_t = np.array(self.A.evalf(subs={
                'x': x,
                'y': y,
                'theta': theta,
                's': self.s,
                'dt': self.d_t
        })).astype(float) + self.x_last
    
        
        self.x_last = x.copy()
        
        return x
    
    def x_t_noise(self, x, dubins_path, i=0) -> np.ndarray:
        
        while i < len(self.s):
            x.append(
                np.array(
                    self.A.evalf(
                        subs={
                            'x': x[-1][0],
                            'y': x[-1][1],
                            'theta': x[-1][2],
                            's': self.s[i],
                            'u': self.dubins_heading(x[-1][0], x[-1][1])[0],
                            'dt': self.d_t
                        }
                    )
                ).astype(float).T[0]
            )
            # consume the list
            return self.x_t_noise(x, dubins_path, i+1)
        return x
        


LTI(s=np.ones((240)), dt=0.5, x0=configurations[0], dubins_path=configurations).x_t_noise(x=[configurations[0]], dubins_path=configurations,)

