import numpy as np 
import crocoddyl 


class DifferentialActionModelCliff(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, isTerminal=False):
        nq = 2 
        nv = 2 
        nx = nv + nq 
        ndx = nx 
        nu = 2 
        state =  crocoddyl.StateVector(nx)
        crocoddyl.DifferentialActionModelAbstract.__init__(self, state, nu, ndx)
        self.g = np.array([0. -9.81])
        self.isTerminal = isTerminal
        self.mass = 1. 

    def _running_cost(self, x, u):
        cost = 0.1/((.1*x[1]**2 + .1)**10) + u[0]**2 + .01*u[1]**2 
        return cost

    def _terminal_cost(self, x, u):
        cost = 100*(x[0]-10)**2 + 10*x[1]**2 + 10*x[2]**2 + 10*x[3]**2  
        return cost 
     
    def calc(self, data, x, u=None):
        if u is None: 
            u = np.zeros(self.nu)
        p =x[:2]
        v =x[2:]
        f = u.copy()

        data.xout = (1/self.mass)*u + self.g # this has the acceleration output
        if self.isTerminal: 
            data.cost = self._terminal_cost(x,u) 
        else:
            data.cost = self._running_cost(x,u)
            
     
