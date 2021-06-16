import numpy as np 
import crocoddyl 
LINE_WIDTH = 100 

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

        data.xout = (1/self.mass)*u + self.g # this has the acceleration output
        if self.isTerminal: 
            data.cost = self._terminal_cost(x,u) 
        else:
            data.cost = self._running_cost(x,u)
    
    def calcDiff(self, data, x, u=None):
        # Advance user might implement the derivatives
        pass
            
     
if __name__ =="__main__":
    print(" Testing Point Mass Cliff with DDP ".center(LINE_WIDTH, '#'))

    
    cliff_diff_running =  crocoddyl.DifferentialActionModelNumDiff(DifferentialActionModelCliff(), True)
    cliff_diff_terminal = crocoddyl.DifferentialActionModelNumDiff(DifferentialActionModelCliff(isTerminal=True), True) 
    print(" Constructing differential models completed ".center(LINE_WIDTH, '-'))
    dt = 0.01 
    T = 300 
    x0 = np.zeros(4) 
    MAX_ITER = 1000
    cliff_running = crocoddyl.IntegratedActionModelEuler(cliff_diff_running, dt) 
    cliff_terminal = crocoddyl.IntegratedActionModelEuler(cliff_diff_terminal, dt) 
    print(" Constructing integrated models completed ".center(LINE_WIDTH, '-'))

    problem = crocoddyl.ShootingProblem(x0, [cliff_running]*T, cliff_terminal)
    print(" Constructing shooting problem completed ".center(LINE_WIDTH, '-'))
    
    ddp = crocoddyl.SolverFDDP(problem)
    print(" Constructing DDP solver completed ".center(LINE_WIDTH, '-'))
    ddp.setCallbacks([
    crocoddyl.CallbackLogger(),
    crocoddyl.CallbackVerbose()
    ])
    xs = [x0]*(T+1)
    us = [np.zeros(2)]*T
    print(ddp.solve(xs,us, MAX_ITER))

    # if converged:
    #     print(" DDP solver has CONVERGED ".center(LINE_WIDTH, '-'))

    

  



