import numpy as np 
import scipy.linalg as scl

from crocoddyl import SolverAbstract

VERBOSE = False  
def rev_enumerate(l):
    return reversed(list(enumerate(l)))


def raiseIfNan(A, error=None):
    if error is None:
        error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A)) > 1e30):
        raise error




class ProcessRiskSensitiveSolver(SolverAbstract):
    def __init__(self, shootingProblem, measurementModel, sensitivity):
        SolverAbstract.__init__(self, shootingProblem)

        # Change it to true if you know that datas[t].xnext = xs[t+1]
        self.isFeasible = False
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

    def models(self):
        mod = [m for m in self.problem.runningModels]
        mod += [self.problem.terminalModel]
        return mod 

    def calc(self):
        try:
            if VERBOSE: print("problem.calcDiff Started")
            self.problem.calc(self.xs, self.us)
            self.cost = self.problem.calcDiff(self.xs, self.us)
            if VERBOSE: print("problem.calcDiff completed with cost %s Now going into measurement model calc "%self.cost)

            for t, (d, measurementMod, mdata) in enumerate(zip(self.problem.runningDatas,
                                            self.measurement.measurementModels, 
                                            self.measurement.runningDatas)):
                measurementMod.calcDiff(d, mdata, self.xs[t], self.us[t])
                self.estimatorGain(t, d, mdata)
            if VERBOSE: print("measurement model cal and estimator gains are completed !!!!!")
        except:
            raise BaseException("Calc Failed !")
        

    def computeDirection(self, recalc=True):
        """ Compute the descent direction dx,dx.

        :params recalc: True for recalculating the derivatives at current state and control.
        :returns the descent direction dx,du and the dual lambdas as lists of
        T+1, T and T+1 lengths.
        """
        if recalc:
            if VERBOSE: print("Going into Calc")
            self.calc()
        try:
            self.backwardPass()
        except:
            raise BaseException("Backward Pass Failed !")

    def tryStep(self, stepLength):
        """ Rollout the system with a predefined step length.

        :param stepLength: step length
        """
        self.forwardPass(stepLength)
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
        self.setCandidate(init_xs, init_us, True)
        if VERBOSE: print("solve setCandidate works just fine ")
        assert (self.xs[-1] == init_xs[-1]).all()
        assert (self.us[-1] == init_us[-1]).all()
        assert self.isFeasible == isFeasible 
        if VERBOSE: print("assert that solver stored values are the same as the input, also works just fine ")
        self.n_little_improvement = 0

        self.x_reg = regInit if regInit is not None else self.regMin
        self.u_reg = regInit if regInit is not None else self.regMin
        if VERBOSE: print("adds x_Reg and u_Reg also works good! now what ? ")
        for i in range(maxiter):
            recalc = True
            while True:
                try: 
                    if VERBOSE: print("Going into compute direction")
                    self.computeDirection(recalc=recalc)
                except BaseException:
                    print('computing direction at iteration %s failed, increasing regularization ' % i)
                    recalc = False
                    self.increaseRegularization()
                    if self.x_reg == self.regMax:
                        return self.xs, self.us, False
                    else:
                        continue
                break

            if i == 0:
                self.cost = 1.e45

            for a in self.alphas:
                try:
                    self.dV = self.tryStep(a)
                except:
                    print('Try step failed ')
                    continue

                if self.dV > 0.:
                    # Accept step
                    #TODO: find a better criteria to accept the step 
                    self.setCandidate(self.xs_try, self.us_try, True)
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
                print('Solver converged ')
                return self.xs, self.us, True
            
            if self.n_little_improvement == 6:
                print(' solver converged with little improvements in the last 6 steps ')
                return self.xs, self.us, True
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
        xtry, utry = self.xs_try, self.us_try
        ctry = 0
        for t, (m, d) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            self.us_try[t] = us[t] + stepLength*self.k[t] + \
                self.K[t].dot(m.state.diff(self.xs_try[t], xs[t]))
            
            with np.warnings.catch_warnings():
                np.warnings.simplefilter(warning)
                m.calc(d, self.xs_try[t], self.us_try[t])
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


    def computeScalarTerms(self):
        pass 

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
        
            gaps = np.zeros(model.state.ndx)
            invWt = ymodel.sn  - self.sigma *ymodel.sd.T.dot(self.St[t+1]).dot(ymodel.sd)
            try: 
                self.Wt[t][:,:] = np.linalg.inv(invWt) 
            except:
                raise BaseException("Wt inversion failed at t = %s"%t)

            # more auxilary terms for the control optimization 
            cwcT = ymodel.sd.dot(self.Wt[t]).dot(ymodel.sd.T) 
            ScwcT = self.St[t+1].dot(cwcT)
            sigScwcT = self.sigma *ScwcT
            sigScwcTS = self.sigma * ScwcT.dot(self.St[t+1]) 

            # control optimization recursions 
            self.Pt[t][:,:] = data.Luu + data.Fu.T.dot(self.St[t+1]+ sigScwcTS).dot(data.Fu)
            self.Tt[t][:,:] = data.Fu.T.dot(self.S[t+1]+ sigScwcTS).dot(data.Fx) 
            self.pt[t][:] = data.Lu + data.Fu.T.dot(np.eye(model.state.ndx) + sigScwcT).dot(self.st[t+1]) 
            self.pt[t][:] += data.Fu.T.dot(np.eye(model.state.ndx) + sigScwcT).dot(self.S[t+1]).dot(gaps)

            # solving for the control 
            try:
                Lb = scl.cho_factor(self.Pt[t], lower=True)
                self.k[t][:] = - scl.cho_solve(Lb, self.pt[t])
                self.K[t][:, :] = - scl.cho_solve(Lb, self.Tt[t][:,:])
            except:
                raise BaseException('choelskey error')

            # Aux terms 
            A_BK = data.Fx + data.Fu.dot(self.K[t])
            SBk_Sf = self.S[t+1].dot(data.Fu.dot(self.k[t]) + gaps)


            self.S[t][:,:] = data.Lxx + self.K[t].T.dot(data.Luu).dot(self.K[t])
            self.S[t][:,:] += A_BK.T.dot(sigScwcTS).dot(A_BK)

            self.s[t][:] = data.Lx + self.K[t].T.dot(data.Luu.dot(self.k[t])+data.Lu)
            self.s[t][:] += A_BK.T.dot(SBk_Sf + self.s[t+1])
            self.s[t][:] += A_BK.T.dot(sigScwcT).dot(SBk_Sf+ self.s[t+1])

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
        self.Wt = [np.zeros([y.np, y.np]) for y in self.measurement.measurementModels] 
        self.Pt = [np.zeros([m.nu, m.nu]) for m in self.problem.runningModels]
        self.pt = [np.zeros(m.nu) for m in self.problem.runningModels]
        self.Tt = [np.zeros([m.nu, m.state.ndx]) for m in self.problem.runningModels]
        # Value function approximations 
        self.St = [np.zeros([m.state.ndx, m.state.ndx]) for m in self.models()]
        self.st = [np.zeros(m.state.ndx) for m in self.models()]
        self.Ft = [np.nan for _ in self.models()]
