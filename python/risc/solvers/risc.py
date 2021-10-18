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
    def __init__(self, shootingProblem, measurementModel, sensitivity, withMeasurement=False):
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
        self.n_little_improvement = 0 
        self.withMeasurement = withMeasurement 
        self.gap_tolerance = 1.e-7
        self.withGaps = False 

        if self.withMeasurement:
            self.allocateDataMeasurement() 
        else:
            self.allocateData()
        print("Data allocated succesfully")

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
                self.filterPass() 
                if VERBOSE: print("measurement model cal and estimator gains are completed !!!!!")
        except:
            raise BaseException("Calc Failed !")
    
    def computeGaps(self):
        could_be_feasible = True 
        # Gap store the state defect from the guess to feasible (rollout) trajectory, i.e.
        #   gap = x_rollout [-] x_guess = DIFF(x_guess, x_rollout)
        m = self.problem.runningModels[0]
        if self.withMeasurement:
            self.fs[0][:m.state.ndx] = m.state.diff(self.xs[0], self.problem.initialState)
            self.fs[0][m.state.ndx:] = m.state.diff(self.xs[0], self.problem.initialState)
        else:
            self.fs[0] = m.state.diff(self.xs[0], self.problem.initialState)
            
        for i, (m, d, x) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas, self.xs[1:])):
            if self.withMeasurement:
                self.fs[i + 1][:m.state.ndx] = m.state.diff(x, d.xnext)
                self.fs[i + 1][m.state.ndx:] = m.state.diff(x, d.xnext)
            else:
                self.fs[i + 1] = m.state.diff(x, d.xnext)
        
        for i in range(self.problem.T+1): 
            if np.linalg.norm(self.fs[i], ord=np.inf) >= self.gap_tolerance:
                could_be_feasible = False
                break
        self.isFeasible = could_be_feasible 

    def filterPass(self):
        """ computes the extended kalman filter along the predetermined trajectory """
        for t, (d, ymodel, ydata) in enumerate(zip(self.problem.runningDatas,
                                                self.measurement.runningModels, 
                                                self.measurement.runningDatas)):
            ymodel.calcDiff(d, ydata, self.xs[t], self.us[t])
            term = ydata.dx.dot(self.Covariance[t]).dot(ydata.dx)
            term += ymodel.md.dot(ymodel.mn).dot(ymodel.md)

            term2 = d.Fx.dot(self.Covariance[t]).dot(ydata.dx.T)

            Lb = scl.cho_factor(term.T , lower=True)
            G_transpose = scl.cho_solve(Lb, term2.T)

            self.G[t][:,:] = G_transpose.T 

            A_GF = d.Fx - self.G[t].dot(ydata.dx)
            self.Covariance[t+1][:,:] = A_GF.dot(self.Covariance[t]).dot(A_GF.T) 
            self.Covariance[t+1][:,:] += ymodel.sd.dot(ymodel.sn).dot(ymodel.sd.T)
            self.Covariance[t+1][:,:] += self.G[t].dot(ymodel.md).dot(ymodel.mn).dot(ymodel.md.T).dot(self.G[t].T)



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
                if self.withMeasurement:
                    self.backwardPassMeasurement()
                else:
                    self.backwardPass()
                 
            except BaseException:
                print('computing direction at iteration %s failed, increasing regularization ' % (self.iter+1))
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
        if self.withMeasurement:
            self.fs[0] = np.zeros(2*self.problem.runningModels[0].state.ndx)
        else:
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
            if self.withMeasurement:
                self.fs[t+1] = np.zeros(2*m.state.ndx)
            else:
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



    def backwardPassMeasurement(self):
        # initialize recursions 
        m = self.problem.terminalModel
        self.st[-1][:m.state.ndx] = self.problem.terminalData.Lx
        self.St[-1][:m.state.ndx,:m.state.ndx] = self.problem.terminalData.Lxx
        if VERBOSE: print(" terminal Value function constructed ".center(LINE_WIDTH,"-"))
        for t, (model, data, ymodel, ydata) in rev_enumerate(zip(self.problem.runningModels,
                                                         self.problem.runningDatas,
                                                         self.measurement.runningModels,  
                                                         self.measurement.runningDatas)):

            # matrix that augments state and measurement covariances   
            self.noise[t][:ymodel.np, :ymodel.np] = ymodel.sn.copy()
            self.noise[t][ymodel.np:, ymodel.np:] = ymodel.mn.copy()
            if VERBOSE: print("noise constructed") 
            self.A[t][:model.state.ndx, :model.state.ndx] = data.Fx 
            self.A[t][model.state.ndx:, :model.state.ndx] = self.G[t].dot(ydata.dx) 
            self.A[t][model.state.ndx:, model.state.ndx:] =data.Fx -  self.G[t].dot(ydata.dx)  
            if VERBOSE: print("A[t] constructed")
            self.B[t][:model.state.ndx,:] = data.Fu
            self.B[t][model.state.ndx:,:] = data.Fu 
            if VERBOSE: print("B[t] constructed")
            self.C[t][:ymodel.np,:ymodel.np] = ymodel.sd 
            self.C[t][ymodel.np:,ymodel.np:] = ymodel.md 
            if VERBOSE: print("C[t] constructed")
            self.Q[t][:model.state.ndx,:model.state.ndx] = data.Lxx   
            self.q[t][:model.state.ndx] = data.Lx  
            self.O[t][:,:model.state.ndx] = data.Lxu.T  
            if VERBOSE: print("cost constructed")
            if VERBOSE: print(" extended system constructed ".center(LINE_WIDTH,"-"))

            # compute this Wt term 
            invWt = np.linalg.inv(self.noise[t])  - self.sigma *self.C[t].T.dot(self.St[t+1]).dot(self.C[t])
            if VERBOSE: print(" invW constructed ".center(LINE_WIDTH,"-"))
            try: 
                self.Wt[t][:,:] = np.linalg.inv(invWt) 
                if VERBOSE: print(" W computed ".center(LINE_WIDTH,"-"))
            except:
                raise BaseException("Wt inversion failed at t = %s"%t)
            

            # more auxiliary terms for the control optimization 
            cwcT = self.C[t].dot(self.Wt[t]).dot(self.C[t].T) 
            ScwcT = self.St[t+1].dot(cwcT)
            sigScwcT = self.sigma *ScwcT
            sigScwcTS = sigScwcT.dot(self.St[t+1])
            S_sigScwcTS = self.St[t+1] + sigScwcTS
            I_sigScwcT = np.eye(2*model.state.ndx) + sigScwcT
            s_Sf = self.st[t+1] + self.St[t+1].dot(self.fs[t+1])
            if VERBOSE: print(" Auxiliary terms constructed ".center(LINE_WIDTH,"-"))

            # control optimization recursions 
            self.Pt[t][:,:] = data.Luu + self.B[t].T.dot(S_sigScwcTS).dot(self.B[t]) 
            self.Tt[t][:,:] = self.O[t] + self.B[t].T.dot(S_sigScwcTS).dot(self.A[t])
            self.pt[t][:] = data.Lu + self.B[t].T.dot(I_sigScwcT).dot(s_Sf)
            if VERBOSE: print(" Controls Terms ".center(LINE_WIDTH,"-"))
            # solving for the control 
            try:
                Lb = scl.cho_factor(self.Pt[t] + self.u_reg*np.eye(model.nu), lower=True)
                # Lb = scl.cho_factor(self.Pt[t], lower=True)
                self.k[t][:] = scl.cho_solve(Lb, -self.pt[t])
                self.Kxxh[t][:, :] = scl.cho_solve(Lb, -self.Tt[t])
            except:
                print("computing controls at node %s failed!"%t)

            if VERBOSE: print(" Controls Optimized ".center(LINE_WIDTH,"-"))
            # Aux terms 
            self.Khat[t][:,model.state.ndx:] = self.Kxxh[t][:, :model.state.ndx] + self.Kxxh[t][:, model.state.ndx:]
            self.K[t][:,:] = self.Kxxh[t][:, :model.state.ndx] + self.Kxxh[t][:, model.state.ndx:]
            A_BK = self.A[t] + self.B[t].dot(self.Khat[t])
            # value function hessian  
            self.St[t][:,:] = self.Q[t] + self.Khat[t].T.dot(data.Luu).dot(self.Khat[t])
            self.St[t][:,:] += self.Khat[t].T.dot(self.O[t]) + self.O[t].T.dot(self.Khat[t])
            self.St[t][:,:] +=  A_BK.T.dot(S_sigScwcTS).dot(A_BK)
            self.St[t][:, :] += self.x_reg*np.eye(2*model.state.ndx)
            if VERBOSE: print(" Value function hessian ".center(LINE_WIDTH,"-"))
            self.St[t][:,:] = .5*(self.St[t]+ self.St[t].T) #symmetric by construction, just make sure here
            # value fcn gradient 
            self.st[t][:] =  self.q[t] + self.Khat[t].T.dot(data.Luu.dot(self.k[t])+data.Lu) + self.O[t].T.dot(self.k[t])
            self.st[t][:] += A_BK.T.dot(I_sigScwcT).dot(self.st[t+1])
            self.st[t][:] += A_BK.T.dot(S_sigScwcTS).dot(self.B[t].dot(self.k[t]) + self.fs[t+1]) 
            if VERBOSE: print(" Value function gradient ".center(LINE_WIDTH,"-"))
    
    def allocateDataMeasurement(self):
        self.xs_try = [self.problem.x0] + [np.nan] * self.problem.T
        self.us_try = [np.nan] * self.problem.T 

        self.Kxxh = [np.zeros([m.nu, 2*m.state.ndx]) for m in self.problem.runningModels]
        self.Khat = [np.zeros([m.nu, 2*m.state.ndx]) for m in self.problem.runningModels]
        self.K = [np.zeros([m.nu, m.state.ndx]) for m in self.problem.runningModels]
        self.k = [np.zeros([m.nu]) for m in self.problem.runningModels]

        self.Wt = [np.zeros([y.np + y.nm, y.np + y.nm]) for y in self.measurement.runningModels] 

        self.Pt = [np.zeros([m.nu, m.nu]) for m in self.problem.runningModels]
        self.pt = [np.zeros(m.nu) for m in self.problem.runningModels]
        self.Tt = [np.zeros([m.nu, 2*m.state.ndx]) for m in self.problem.runningModels]
        # Value function approximations 
        self.St = [np.zeros([2*m.state.ndx, 2*m.state.ndx]) for m in self.models()]
        self.st = [np.zeros(2*m.state.ndx) for m in self.models()]
        self.Ft = [np.nan for _ in self.models()]
        # The extended Dynamics & Cost  
        self.fs = [np.zeros(2*m.state.ndx) for m in self.models()] 
        self.A = [np.zeros([2*m.state.ndx, 2*m.state.ndx]) for m in self.models()]
        self.B = [np.zeros([2*m.state.ndx, m.nu]) for m in self.models()]
        self.C = [np.zeros([2*m.state.ndx, y.np + y.nm]) for m,y in zip(self.problem.runningModels, self.measurement.runningModels)]
        self.noise = [np.zeros([y.np + y.nm, y.np + y.np]) for _,y in zip(self.problem.runningModels, self.measurement.runningModels)]
        self.Q = [np.zeros([2*m.state.ndx, 2*m.state.ndx]) for m in self.models()]
        self.q = [np.zeros(2*m.state.ndx) for m in self.models()]
        self.O = [np.zeros([m.nu, 2*m.state.ndx]) for m in self.models()]
        # The Kalman Filter 
        self.G = [np.zeros([m.state.ndx, y.ny]) for m,y in zip(self.problem.runningModels, self.measurement.runningModels)] 
        self.Covariance = [np.zeros([m.state.ndx, m.state.ndx]) for m in self.models()] 
        ymodel = self.measurement.runningModels[0] 
        self.Covariance[0][:,:] = ymodel.sd.dot(ymodel.sn).dot(ymodel.sd.T)