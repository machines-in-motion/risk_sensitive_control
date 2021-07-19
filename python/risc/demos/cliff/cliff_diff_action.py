import numpy as np 
import crocoddyl 
import matplotlib.pyplot as plt 
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
        self.g = np.array([0. , -9.81])
        self.isTerminal = isTerminal
        self.mass = 1. 

    def _running_cost(self, x, u):
        cost = 0.1/((.1*x[1] + .1)**10) + u[0]**2 + .01*u[1]**2 
        return cost

    def _terminal_cost(self, x, u):
        cost = 1000*((x[0]-10.)**2) + 1000*(x[1]**2) + 1000*(x[2]**2) + 1000*(x[3]**2)  
        return cost 
     
    def calc(self, data, x, u=None):
        if u is None: 
            u = np.zeros(self.nu)

        data.xout = (1/self.mass)*u + self.g # this has the acceleration output
        if self.isTerminal: 
            data.cost = self._terminal_cost(x,u) 
        else:
            data.cost = self._running_cost(x,u)

        # data.r = None # residuals I think, Must be crucial for derivative computation, must check it  
    
    def calcDiff(self, data, x, u=None):
        # Advance user might implement the derivatives
        Fx = np.zeros([2,4]) 
        Fu = np.zeros([2,2])
        Fu[0,0] = 1./self.mass 
        Fu[1,1] = 1./self.mass 
        Lx = np.zeros([4])
        Lu = np.zeros([2])
        Lxx = np.zeros([4,4])
        Luu = np.zeros([2,2])
        Lxu = np.zeros([4,2])
        if self.isTerminal:
            Lx[0] = 2000.*(x[0]-10)
            Lx[1] = 2000.*x[1]
            Lx[2] = 2000.*x[2]
            Lx[3] = 2000.*x[3]     
            Lxx[0,0] = 2000. 
            Lxx[1,1] = 2000. 
            Lxx[2,2] = 2000. 
            Lxx[3,3] = 2000. 
        else:
            Lx[1] = - (10.**10) /((x[1]+10.)**11)
            Lu[0] = 2.*u[0] 
            Lu[1] = 0.02 * u[1]
            Lxx[1,1] = 0.11/((.1*x[1]+1.)**12)
            Luu[0,0] = 2. 
            Luu[1,1] = 0.02

        data.Fx = Fx.copy()
        data.Fu = Fu.copy()
        data.Lx = Lx.copy()
        data.Lu = Lu.copy()
        data.Lxx = Lxx.copy()
        data.Luu = Luu.copy()
        data.Lxu = Lxu.copy()


            
     
if __name__ =="__main__":
    print(" Testing Point Mass Cliff with DDP ".center(LINE_WIDTH, '#'))
    cliff_diff_running =  DifferentialActionModelCliff()
    cliff_diff_terminal = DifferentialActionModelCliff(isTerminal=True)
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
    
    ddp = crocoddyl.SolverDDP(problem)
    print(" Constructing DDP solver completed ".center(LINE_WIDTH, '-'))
    ddp.setCallbacks([
    crocoddyl.CallbackLogger(),
    crocoddyl.CallbackVerbose()
    ])
    xs = [x0]*(T+1)
    us = [np.zeros(2)]*T
    converged = ddp.solve(xs,us, MAX_ITER)
    x =[]
    y =[]
    vx =[]
    vy =[]
    time_array = dt*np.arange(T+1)
    for xi in ddp.xs:
        x += [xi[0]]
        y += [xi[1]]
        vx += [xi[2]]
        vy += [xi[3]]
    if converged:
        print(" DDP solver has CONVERGED ".center(LINE_WIDTH, '-'))
        plt.figure("trajectory plot")
        plt.plot(x,y)

        plt.figure("velocity plots")
        plt.plot(time_array, vx)
        plt.plot(time_array, vy)
        plt.show()


    

  



