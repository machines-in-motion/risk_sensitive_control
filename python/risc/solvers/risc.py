import numpy as np 
import scipy.linalg as scl

from crocoddyl import SolverAbstract

LINE_WIDTH = 100 

VERBOSE = False  
def rev_enumerate(l):
    return reversed(list(enumerate(l)))


def raiseIfNan(A, error=None):
    if error is None:
        error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A)) > 1e30):
        raise error




class RiskSensitiveSolver(SolverAbstract):
    def __init__(self, shootingProblem, measurementModel, sensitivity):
        SolverAbstract.__init__(self, shootingProblem)

        # Change it to true if you know that datas[t].xnext = xs[t+1]
        self.wasFeasible = False
        self.alphas = [2**(-n) for n in range(10)]
        self.x_reg = 0
        self.u_reg = 0
        self.regFactor = 10
        self.regMax = 1e9
        self.regMin = 1e-9
        self.th_step = .5
        self.th_stop = 1.e-9 
        self.measurement = measurementModel 
        self.sigma = sensitivity
        self.allocateData()
        print("Data allocated succesfully")
        self.n_little_improvement = 0 
        self.withMeasurement = False 
        self.gap_tolerance = 1.e-7
        self.withGaps = False 


    def models(self):
        mod = [m for m in self.problem.runningModels]
        mod += [self.problem.terminalModel]
        return mod 

    def calc(self):
        try:
            if VERBOSE: print("problem.calcDiff Started")
            # if self.iter == 0:
            self.problem.calc(self.xs, self.us)
                # self.firstIter = False 
            self.cost = self.problem.calcDiff(self.xs, self.us)
            if VERBOSE: print("problem.calcDiff completed with cost %s Now going into measurement model calc "%self.cost)
                
            if self.withGaps:
                self.computeGaps()
        
            if self.withMeasurement:
                pass 
                # for t, (d, measurementMod, mdata) in enumerate(zip(self.problem.runningDatas,
                #                                 self.measurement.measurementModels, 
                #                                 self.measurement.runningDatas)):
                #     measurementMod.calcDiff(d, mdata, self.xs[t], self.us[t])
                #     self.estimatorGain(t, d, mdata)
                # if VERBOSE: print("measurement model cal and estimator gains are completed !!!!!")
        except:
            raise BaseException("Calc Failed !")
    
    def computeGaps(self):
        could_be_feasible = True 
        # Gap store the state defect from the guess to feasible (rollout) trajectory, i.e.
        #   gap = x_rollout [-] x_guess = DIFF(x_guess, x_rollout)
        self.fs[0] = self.problem.runningModels[0].state.diff(self.xs[0], self.problem.initialState)
            
        for i, (m, d, x) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas, self.xs[1:])):
            self.fs[i + 1] = m.state.diff(x, d.xnext)
        
        for i in range(self.problem.T+1): 
            if np.linalg.norm(self.fs[i], ord=np.inf) >= self.gap_tolerance:
                could_be_feasible = False
                break
        self.isFeasible = could_be_feasible 


    def computeDirection(self, recalc=True):
        """ Compute the descent direction dx,dx.

        :params recalc: True for recalculating the derivatives at current state and control.
        :returns the descent direction dx,du and the dual lambdas as lists of
        T+1, T and T+1 lengths.
        """

        if VERBOSE: print("Going into Calc".center(LINE_WIDTH,"-"))
        self.calc()
        while True:
            try:
                if VERBOSE: print("Going into Backward Pass".center(LINE_WIDTH,"-"))
                self.backwardPass()
                 
            except BaseException:
                print('computing direction at iteration %s failed, increasing regularization ' % i)
                self.increaseRegularization()
                if self.x_reg == self.regMax:
                    return False
                else:
                    continue
            break
        return True 

    def tryStep(self, stepLength):
        """ Rollout the system with a predefined step length.

        :param stepLength: step length
        """
        if self.withGaps:
            self.forwardPass(stepLength)
        else:
            self.forwardPassNoGaps(stepLength)
        return self.cost - self.cost_try

    def expectedImprovement(self):
        return np.array([0.]), np.array([0.])
        

    def stoppingCriteria(self):
        # for now rely on feedforward norm 
        knormSquared = [ki.dot(ki) for ki in self.k]
        knorm = np.sqrt(np.array(knormSquared))
        return knorm
        
    def solve(self, maxiter=100, init_xs=None, init_us=None, isFeasible=False, regInit=None):
        """ Nonlinear solver iterating over the solveQP.

        Compute the optimal xopt,uopt trajectory as lists of T+1 and T terms.
        And a boolean describing the success.
        :param maxiter: Maximum allowed number of iterations
        :param init_xs: Initial state
        :param init_us: Initial control
        """
        
        # if not accounting for gaps rollout initial trajectory 
        if not self.withGaps:
            init_xs = self.problem.rollout(init_us)
            isFeasible = True
        
        # set solver.xs and solver.us and solver.isFeasible   
        self.setCandidate(init_xs, init_us, isFeasible)
        
        self.n_little_improvement = 0
        # set regularization values 
        self.x_reg = regInit if regInit is not None else self.regMin
        self.u_reg = regInit if regInit is not None else self.regMin

        for i in range(maxiter):
            recalc = True # flag to recalculate dynamics & derivatives 
            # backward pass and regularize 
            backwardFlag = self.computeDirection(recalc=recalc)

            if not backwardFlag:
                # if backward pass fails after all regularization
                print(' Failed to compute backward pass at maximum regularization '.center(LINE_WIDTH,'#')) 
                return self.xs, self.us, False

           
            for a in self.alphas:
                try:
                    self.dV = self.tryStep(a)
                except:
                    print('Try step failed ')
                    continue

                if self.dV > 0.:
                    # Accept step
                    #TODO: find a better criteria to accept the step 
                    self.setCandidate(self.xs_try, self.us_try, self.isFeasible)
                    self.cost = self.cost_try
                    break
                # else:
                #     self.n_little_improvement += 1  
            if a > self.th_step:
                self.decreaseRegularization()
            if a == self.alphas[-1] :
                self.n_little_improvement += 1 
                self.increaseRegularization()
                if self.x_reg == self.regMax:
                    return self.xs, self.us, False
            # else:
            #     self.n_little_improvement = 0
            self.stepLength = a
            self.iter = i
            self.stop = sum(self.stoppingCriteria())
            if self.callback is not None:
                [c(self) for c in self.callback]

            if  self.stop < self.th_stop:
                print('Feedforward Norm %s, Solver converged '%self.stop)
                return self.xs, self.us, True
            
            if self.n_little_improvement == 10:
                print(' solver converged with little improvements in the last 6 steps ')
                return self.xs, self.us, self.isFeasible
        # Warning: no convergence in max iterations
        print('max iterations with no convergance')
        return self.xs, self.us, self.isFeasible 

    def increaseRegularization(self):
        self.x_reg *= self.regFactor
        if self.x_reg > self.regMax:
            self.x_reg = self.regMax
        self.u_reg = self.x_reg

    def decreaseRegularization(self):
        self.x_reg /= self.regFactor
        if self.x_reg < self.regMin:
            self.x_reg = self.regMin
        self.u_reg = self.x_reg

    def forwardPass(self, stepLength, warning='error'):
        xs, us = self.xs, self.us
        ctry = 0
        

        for t, (m, d) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            # handle gaps 
            if self.isFeasible or stepLength == 1:
                self.xs_try[t] = d.xnext.copy()
            else:
                self.xs_try[t] = m.state.integrate(d.xnext, self.fs[t] * (stepLength - 1))
            # update control 
            self.us_try[t] = us[t] + stepLength*self.k[t] + \
                self.K[t].dot(m.state.diff(xs[t], self.xs_try[t]))
            
            with np.warnings.catch_warnings():
                np.warnings.simplefilter(warning)
                m.calc(d, self.xs_try[t], self.us_try[t])
            # update state 
            self.xs_try[t + 1] = d.xnext.copy()  # not sure copy helpful here.
            ctry += d.cost
            raiseIfNan([ctry, d.cost], BaseException('forward error'))
            raiseIfNan(self.xs_try[t + 1], BaseException('forward error'))
        with np.warnings.catch_warnings():
            np.warnings.simplefilter(warning)
            self.problem.terminalModel.calc(
                self.problem.terminalData, self.xs_try[-1])
            ctry += self.problem.terminalData.cost
        raiseIfNan(ctry, BaseException('forward error'))
        self.cost_try = ctry
        return self.xs_try, self.us_try, ctry

    def forwardPassNoGaps(self, stepLength, warning='error'):
        ctry = 0
        self.xs_try[0] = self.xs[0].copy()
        self.fs[0] = np.zeros(self.problem.runningModels[0].state.ndx)
        for t, (m, d) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            # update control 
            self.us_try[t] = self.us[t] + stepLength*self.k[t] + \
                self.K[t].dot(m.state.diff(self.xs[t], self.xs_try[t]))
            
            with np.warnings.catch_warnings():
                np.warnings.simplefilter(warning)
                m.calc(d, self.xs_try[t], self.us_try[t])
            # update state 
            self.xs_try[t + 1] = d.xnext.copy()  # not sure copy helpful here.
            ctry += d.cost
            self.fs[t+1] = np.zeros(m.state.ndx)
            raiseIfNan([ctry, d.cost], BaseException('forward error'))
            raiseIfNan(self.xs_try[t + 1], BaseException('forward error'))
        with np.warnings.catch_warnings():
            np.warnings.simplefilter(warning)
            self.problem.terminalModel.calc(
                self.problem.terminalData, self.xs_try[-1])
            ctry += self.problem.terminalData.cost
        raiseIfNan(ctry, BaseException('forward error'))
        self.cost_try = ctry
        self.isFeasible = True 
        return self.xs_try, self.us_try, ctry



    def backwardPass(self):
        # initialize recursions 
        self.st[-1][:] = self.problem.terminalData.Lx
        self.St[-1][:,:] = self.problem.terminalData.Lxx
        # self.Ft[-1] = self.problem.terminalData.cost 
        #TODO: check the recursions once more
        # iterate backwards 
        for t, (model, data, ymodel, ydata) in rev_enumerate(zip(self.problem.runningModels,
                                                         self.problem.runningDatas,
                                                         self.measurement.runningModels,  
                                                         self.measurement.runningDatas)):
        
            invWt = np.linalg.inv(ymodel.sn)  - self.sigma *ymodel.sd.T.dot(self.St[t+1]).dot(ymodel.sd)
            if VERBOSE: print(" invW constructed ".center(LINE_WIDTH,"-"))
            try: 
                self.Wt[t][:,:] = np.linalg.inv(invWt) 
                if VERBOSE: print(" W computed ".center(LINE_WIDTH,"-"))
            except:
                raise BaseException("Wt inversion failed at t = %s"%t)
            

            # more auxiliary terms for the control optimization 
            cwcT = ymodel.sd.dot(self.Wt[t]).dot(ymodel.sd.T) 
            ScwcT = self.St[t+1].dot(cwcT)
            sigScwcT = self.sigma *ScwcT
            sigScwcTS = sigScwcT.dot(self.St[t+1]) 
            S_sigScwcTS = self.St[t+1] + sigScwcTS
            I_sigScwcT = np.eye(model.state.ndx) + sigScwcT
            s_Sf = self.st[t+1] + self.St[t+1].dot(self.fs[t+1])
            if VERBOSE: print(" Auxiliary terms constructed ".center(LINE_WIDTH,"-"))

            # control optimization recursions 
            self.Pt[t][:,:] = data.Luu + data.Fu.T.dot(S_sigScwcTS).dot(data.Fu) 
            if VERBOSE: print(" P[t] constructed ".center(LINE_WIDTH,"-"))
            self.Tt[t][:,:] = data.Lxu.T + data.Fu.T.dot(S_sigScwcTS).dot(data.Fx)
            if VERBOSE: print(" T[t] constructed ".center(LINE_WIDTH,"-"))
            self.pt[t][:] = data.Lu + data.Fu.T.dot(I_sigScwcT).dot(s_Sf)
            if VERBOSE: print(" p[t] constructed ".center(LINE_WIDTH,"-"))
            if VERBOSE: print(" Controls Terms ".center(LINE_WIDTH,"-"))
            # solving for the control 
            try:
                Lb = scl.cho_factor(self.Pt[t] + self.u_reg*np.eye(model.nu), lower=True)
                # Lb = scl.cho_factor(self.Pt[t], lower=True)
                self.k[t][:] = scl.cho_solve(Lb, -self.pt[t])
                self.K[t][:, :] = scl.cho_solve(Lb, -self.Tt[t])
            except:
                pass 
                # raise BaseException('choelskey error')
            if VERBOSE: print(" Controls Optimized ".center(LINE_WIDTH,"-"))
            # Aux terms 
            A_BK = data.Fx + data.Fu.dot(self.K[t])
            

            if VERBOSE: print(" Controls auxiliary terms ".center(LINE_WIDTH,"-"))

            self.St[t][:,:] = data.Lxx + self.K[t].T.dot(data.Luu).dot(self.K[t])
            self.St[t][:,:] += self.K[t].T.dot(data.Lxu.T) + data.Lxu.dot(self.K[t])
            self.St[t][:,:] +=  A_BK.T.dot(S_sigScwcTS).dot(A_BK)
            self.St[t][:, :] += self.x_reg*np.eye(model.state.ndx)
            if VERBOSE: print(" Value function hessian ".center(LINE_WIDTH,"-"))
            self.St[t][:,:] = .5*(self.St[t]+ self.St[t].T)

            self.st[t][:] =  data.Lx + self.K[t].T.dot(data.Luu.dot(self.k[t])+data.Lu) + data.Lxu.dot(self.k[t])
            self.st[t][:] += A_BK.T.dot(I_sigScwcT).dot(self.st[t+1])
            self.st[t][:] += A_BK.T.dot(S_sigScwcTS).dot(data.Fu.dot(self.k[t]) + self.fs[t+1]) 

            if VERBOSE: print(" Value function gradient ".center(LINE_WIDTH,"-"))

    def allocateData(self):
        """  Allocate memory for all variables needed, control, state, value function and estimator.
        """
        # state and control 
        self.xs_try = [self.problem.x0] + [np.nan] * self.problem.T
        self.us_try = [np.nan] * self.problem.T
        # feedforward and feedback 
        self.K = [np.zeros([m.nu, m.state.ndx]) for m in self.problem.runningModels]
        self.k = [np.zeros([m.nu]) for m in self.problem.runningModels]
        # control optimization 
        self.Wt = [np.zeros([y.np, y.np]) for y in self.measurement.runningModels] 
        self.Pt = [np.zeros([m.nu, m.nu]) for m in self.problem.runningModels]
        self.pt = [np.zeros(m.nu) for m in self.problem.runningModels]
        self.Tt = [np.zeros([m.nu, m.state.ndx]) for m in self.problem.runningModels]
        # Value function approximations 
        self.St = [np.zeros([m.state.ndx, m.state.ndx]) for m in self.models()]
        self.st = [np.zeros(m.state.ndx) for m in self.models()]
        self.Ft = [np.nan for _ in self.models()]

        self.fs = [np.zeros(self.problem.runningModels[0].state.ndx)
                     ] + [np.zeros(m.state.ndx) for m in self.problem.runningModels]
