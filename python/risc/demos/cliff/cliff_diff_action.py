"""    """

import numpy as np 
import crocoddyl 


class DifferentialActionModelCliff(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, isTerminal=False):
        nq = 2 
        nv = 2 
        nx = nv + nq 
        ndx = nx 
        nu = 2 
        dt = 0.1
        self.m = 1 
        state =  crocoddyl.StateVector(nx)
        crocoddyl.DifferentialActionModelAbstract.__init__(self, state, nu, ndx)
        self.A = np.array([[1., 0., dt, 0.],
                           [0., 1., 0., dt],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]])
        self.B = (1/self.m) * np.array([[.5 * (dt**2), 0.],
                           [0., .5 * (dt**2)],
                           [dt, 0.],
                           [0., dt]])

        self.g = np.array([0. -9.81])

        self.isTerminal = isTerminal

    def _running_cost(self):
        raise NotImplementedError("Running cost stuff not implemented yet")

    def _terminal_cost(self):
        raise NotImplementedError("Terminal cost stuff not implemented yet")
     
    def calc(self, data, x, u=None):
        if u is None: 
            u = np.zeros(self.nu)
        p =x[:2]
        v =x[2:]
        f = u.copy()
        raise NotImplementedError("Calc method stuff not implemented yet")
     
