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




class RiskSensitiveSolver(SolverAbstract):
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
            self.us_try[t] = us[t] - stepLength*self.k[t] - \
                self.K[t].dot(m.state.diff(xs[t], self.xs_try[t]))
            
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

    def estimatorGain(self, t, data, mdata):
        try:
            invZ = np.linalg.inv(mdata.dx.dot(
                self.cov[t]).dot(mdata.dx.T) + mdata.dd)
        except:
            raise BaseException('Kalman filter error')
        self.filterGains[t][:, :] = data.Fx.dot(
                    self.cov[t]).dot(mdata.dx.T).dot(invZ)
        akf = data.Fx - self.filterGains[t].dot(mdata.dx)
        self.cov[t+1][:, :] =  akf.dot(self.cov[t]).dot(akf.T) + mdata.cc \
            + self.filterGains[t].dot(mdata.dd).dot(self.filterGains[t].T)

    def backwardPass(self):
        # initialize recursions 
        self.st[-1][:self.problem.terminalModel.state.ndx] = self.problem.terminalData.Lx
        self.St[-1][:self.problem.terminalModel.state.ndx, :self.problem.terminalModel.state.ndx] = self.problem.terminalData.Lxx
        #TODO: check the recursions once more
        # iterate backwards 
        for t, (model, data, ymodel, ydata) in rev_enumerate(zip(self.problem.runningModels,
                                                         self.problem.runningDatas,
                                                         self.measurement.runningModels,  
                                                         self.measurement.runningDatas)):

            self.At[t][:model.state.ndx,:model.state.ndx] = data.Fx 
            self.At[t][model.state.ndx:,:model.state.ndx] = self.filterGains[t].dot(ydata.dx)
            self.At[t][model.state.ndx:,model.state.ndx:] = data.Fx - self.filterGains[t].dot(ydata.dx)
            # 
            self.Bt[t][:model.state.ndx, :] = data.Fu 
            self.Bt[t][model.state.ndx:, :] = data.Fu 
            #TODO: check the dimensions of the noise models below  
            self.Ct[t][:model.state.ndx, :ymodel.np] = ymodel.sd 
            self.Ct[t][model.state.ndx:, ymodel.np:] = ymodel.md 
            # 
            self.Qt[t][:model.state.ndx,:model.state.ndx] = data.Lxx 
            self.qt[t][:model.state.ndx] = data.Lx 
            self.Pt[t][:model.state.ndx, :] = data.Lxu 
            # first compute Wt from the document eq.94 
            self.Xit[t][:ymodel.np, :ymodel.np] = ymodel.sn 
            self.Xit[t][ymodel.np:, ymodel.np:] =  ymodel.mn 
            invWt = self.Xit[t] - self.sigma * self.Ct[t].T.dot(self.St[t+1]).dot(self.Ct[t])
            try: 
                self.Wt[t] = np.linalg.inv(invWt) 
            except:
                raise BaseException("Wt inversion failed at t = %s"%t)

            # more auxilary terms for the control optimization 
            cwcT = self.Ct[t].dot(self.Wt[t]).dot(self.Ct[t].T) 
            ScwcT = self.St[t+1].dot(cwcT)
            sigScwcTS = self.sigma * ScwcT.dot(self.St[t+1]) 

            # control optimization recursions 
            self.Gt[t] = data.Luu + self.Bt[t].T.dot(self.St[t+1] + sigScwcTS).dot(self.Bt[t]) 
            self.gt[t] = data.Lu + self.Bt[t].T.dot(self.st[t+1]+ self.sigma * ScwcT.dot(self.st[t+1]))
            self.Ht[t] = self.Bt[t].T.dot(self.St[t+1] + sigScwcTS).dot(self.At[t]) + self.Pt[t].T
            
            # solving for the control 
            try:
                Lb = scl.cho_factor(self.Gt[t], lower=True)
                self.k[t][:] = - scl.cho_solve(Lb, self.gt[t])
                self.K[t][:, :] = - scl.cho_solve(Lb, self.Ht[t][:,:model.ndx]+self.Ht[t][:,model.ndx:])
            except:
                raise BaseException('choelskey error')

            self.Ht_bar[t][:,model.state.ndx:] = self.K[t][:, :]
            # value function approximation 
            P_ASB = self.Pt[t] + self.At[t].T.dot(self.St[t+1]).dot(self.Bt[t])
            HR_HBSB = self.Ht_bar[t].T.dot(data.Luu) +  self.Ht[t].T.dot(self.Bt[t].T).dot(self.St[t+1]).dot(self.Bt[t])
            BH = self.Bt[t].dot(self.Ht_bar[t])
            #
            self.St[t] = self.Qt[t] + (HR_HBSB + 2*P_ASB).dot(self.Ht_bar[t]) + self.At[t].T.dot(self.St[t+1]).dot(self.At[t])
            self.St[t] += self.At[t].T.dot(sigScwcTS).dot(self.At[t]) + 2* self.At[t].T.dot(sigScwcTS).dot(BH) + BH.T.dot(sigScwcTS).dot(BH) 
            #
            self.st[t] = self.qt[t] + (HR_HBSB + P_ASB).dot(self.k[t]) + (self.At[t].T + BH.T).dot(self.st[t+1])
            self.st[t] += (self.At[t] + BH).T.dot(sigScwcTS).dot(self.st[t+1]) + (2*self.At[t] + BH).T.dot(sigScwcTS).dot(self.k[t])


    def allocateData(self):
        """  Allocate memory for all variables needed, control, state, value function and estimator.
        """
        # state and control 
        self.xs_try = [self.problem.x0] + [np.nan] * self.problem.T
        self.us_try = [np.nan] * self.problem.T
        # feedforward and feedback 
        self.K = [np.zeros([m.nu, m.state.ndx]) for m in self.problem.runningModels]
        self.k = [np.zeros([m.nu]) for m in self.problem.runningModels]
        # Auxilary Parameters for the Augmented state space model 
        # defined in equations 78 and 112 
        self.At = [np.zeros([2*m.state.ndx, 2*m.state.ndx]) for m in self.problem.runningModels]
        self.Bt = [np.zeros([2*m.state.ndx, 2*m.nu]) for m in self.problem.runningModels] 
        self.Ct = [np.zeros([2*m.state.ndx, y.ny + y.np])for m, y in zip(self.problem.runningModels, self.measurement.measurementModels)]
        # 
        self.Qt = [np.zeros([2*m.state.ndx, 2*m.state.ndx]) for m in self.models()]
        self.qt = [np.zeros(2*m.state.ndx) for m in self.models()]
        self.Pt = [np.zeros([2*m.state.ndx, m.nu]) for m in self.models()]
        # control optimization 
        self.Xit = [np.zeros([y.np + y.nm, y.np + y.nm]) for y in self.measurement.measurementModels] 
        self.Wt = [np.zeros([y.np + y.nm, y.np + y.nm]) for y in self.measurement.measurementModels] 
        self.Gt = [np.zeros([m.nu, m.nu]) for m in self.problem.runningModels]
        self.gt = [np.zeros(m.nu) for m in self.problem.runningModels]
        self.Ht = [np.zeros([m.nu, 2*m.state.ndx]) for m in self.problem.runningModels]
        self.Ht_bar = [np.zeros([m.nu, 2*m.state.ndx]) for m in self.problem.runningModels]
        # Value function approximations 
        self.St = [np.zeros([2*m.state.ndx, 2*m.state.ndx]) for m in self.models()]
        self.st = [np.zeros(2*m.state.ndx) for m in self.models()]
        self.Ft = [np.nan for m in self.models()]
        # filter parameters 
        self.cov = [np.zeros([m.state.ndx, m.state.ndx]) for m in self.models()]
        self.cov[0] = self.measurement.measurementModels[0].sn.copy()
        self.filterGains = [np.zeros([m.state.ndx, y.ny])for m, y
                            in zip(self.problem.runningModels, self.measurement.measurementModels)]
