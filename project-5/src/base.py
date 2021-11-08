from typing import Callable

import numpy as np


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
        pass

    def _predict(self, *args, **kwargs) -> None:
        pass
    
    def predict(self, *args, **kwargs) -> None:
        self._predict(*args, **kwargs)
        self._record()
    
    def update(self, *args, **kwargs) -> None:
        self._update(*args, **kwargs)
        self._record()



class BaseFilter(RecordableFilter):

    def __init__(self, x: np.ndarray, dim_x: int, dim_y: int, R: np.ndarray, Q: np.ndarray, fx: Callable, hx: Callable, record_variables: list = None,) -> None:
        
        super().__init__(record_variables)
        
        # --
        self.x = x
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

        # ----
        self.I = np.eye(dim_x)
        self.P_posteriori = self.I.copy()
        self.P_priori = self.I.copy()
        self.K = np.zeros((dim_x, dim_y))
