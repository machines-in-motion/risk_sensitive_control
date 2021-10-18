""" This is an older version based on Brahaym's derivations, not complete """

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
        self.th_grad = 1e-12

        self.x_reg = 0
        self.u_reg = 0
        self.regFactor = 10
        self.regMax = 1e9
        self.regMin = 1e-9
        self.th_step = .5
        self.th_acceptNegStep = 2.
        self.measurement = measurementModel 
        self.s = sensitivity
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

        # Risk Sensetive Specific

    def backwardPass(self):

        
        self.Sx[-1][:] = self.problem.terminalData.Lx
        self.Sxx[-1][:, :] = self.problem.terminalData.Lxx
        self.Q0[-1] = self.problem.terminalData.cost
        #TODO: check the recursions once more
        for t, (model, data, mdata) in rev_enumerate(zip(self.problem.runningModels,
                                                         self.problem.runningDatas, 
                                                         self.measurement.runningDatas)):

            kf = self.filterGains[t].dot(mdata.dx)
            assert kf.shape == (model.state.ndx, model.state.ndx)
            akf = data.Fx - kf
            assert akf.shape == (model.state.ndx, model.state.ndx)

            sxxh_a_kf = self.Sxxh[t+1].dot(akf)
            assert sxxh_a_kf.shape == (model.state.ndx, model.state.ndx)
            sxhxh_a_kf = self.Sxhxh[t+1].dot(akf)
            assert sxhxh_a_kf.shape == (model.state.ndx, model.state.ndx)

            kdk = self.filterGains[t].dot(mdata.dd).dot(self.filterGains[t].T)
            assert kdk.shape == (model.state.ndx, model.state.ndx)
            sxxa_sxxh_kf = self.Sxx[t+1].dot(data.Fx) + self.Sxxh[t+1].dot(kf)
            assert sxxa_sxxh_kf.shape == (model.state.ndx, model.state.ndx)
            sxxha_sxhxh_kf = self.Sxxh[t+1].dot(data.Fx) + self.Sxhxh[t+1].dot(kf)
            assert sxxha_sxhxh_kf.shape == (model.state.ndx, model.state.ndx)
            self.H[t][:, :] = data.Luu + data.Fu.T.dot(self.Sxx[t+1] + self.Sxhxh[t+1]
                                                       + self.Sxxh[t+1] + self.Sxxh[t+1].T).dot(data.Fu)\
                + self.s*data.Fu.T.dot((self.Sxx[t+1]+self.Sxxh[t+1]).T).dot(mdata.cc)\
                .dot(self.Sxx[t+1]+self.Sxxh[t+1]).dot(data.Fu)\
                + self.s*data.Fu.T.dot((self.Sxxh[t+1].T+self.Sxhxh[t+1]).T).dot(kdk)\
                .dot(self.Sxxh[t+1].T+self.Sxhxh[t+1]).dot(data.Fu)
            # add regularization to ensure smooth inversion
            if self.u_reg != 0:
                self.H[t][range(model.nu), range(model.nu)] += self.u_reg
            #
            self.g[t][:] = data.Lu + data.Fu.T.dot(self.Sx[t+1] + self.Sxh[t+1])\
                + self.s * data.Fu.T.dot((
                    self.Sxx[t+1]+self.Sxxh[t+1]).T).dot(mdata.cc).dot(self.Sx[t+1])\
                + self.s * data.Fu.T.dot(
                (self.Sxxh[t+1].T+self.Sxhxh[t+1]).T).dot(kdk).dot(self.Sxh[t+1])
            #
            self.Gx[t][:, :] = data.Lxu.T + \
                data.Fu.T.dot(self.Sxx[t+1] + self.Sxxh[t+1]).dot(data.Fx)\
                + data.Fu.T.dot(self.Sxhxh[t+1] + self.Sxxh[t+1]).dot(kf)\
                + self.s * data.Fu.T.dot(
                    (self.Sxx[t+1]+self.Sxxh[t+1]).T).dot(mdata.cc).dot(sxxa_sxxh_kf)\
                + self.s*data.Fu.T.dot(self.Sxxh[t+1]+self.Sxhxh[t+1]).dot(kdk).dot(sxxha_sxhxh_kf)
            #
            self.Gb[t][:, :] = data.Fu.T.dot(self.Sxhxh[t+1] + self.Sxxh[t+1]).dot(akf)\
                + self.s*data.Fu.T.dot((self.Sxx[t+1]+self.Sxxh[t+1]).T).dot(mdata.cc)\
                .dot(self.Sxxh[t+1]).dot(akf)\
                + self.s*data.Fu.T.dot(self.Sxxh[t+1].T+self.Sxhxh[t+1]).dot(kdk)\
                .dot(self.Sxhxh[t+1]).dot(akf)
            # compute feedfoward and feedback terms
            try:
                if self.H[t].shape[0] > 0:
                    Lb = scl.cho_factor(self.H[t], lower=True)
                    self.k[t][:] = scl.cho_solve(Lb, self.g[t])
                    self.K[t][:, :] = scl.cho_solve(Lb, self.Gx[t]+self.Gb[t])
                else:
                    raise BaseException('choelskey error')
            except:
                raise BaseException('backward error')
            # Update Value Function Quadratic Approximation
            # this should be used as an approximation model 
            # self.Q0[t] = self.Q0[t+1] + data.cost - .5*self.g[t].T.dot(self.k[t]) \
            #     + .5*(self.Sxx[t+1].dot(mdata.cc) + self.Sxhxh[t+1].dot(kdk)).trace() \
            #     + .5*self.s*(self.Sx[t+1].T.dot(mdata.cc).dot(self.Sx[t+1])
            #                  + self.Sxh[t+1].T.dot(kdk).dot(self.Sxh[t+1]))
            #
            self.Sx[t][:] = data.Lx + data.Fx.T.dot(self.Sx[t+1]) + kf.T.dot(self.Sxh[t+1]) \
                - self.Gx[t].T.dot(self.k[t]) + self.s * \
                (sxxa_sxxh_kf.T.dot(mdata.cc).dot(self.Sx[t+1])) \
                + self.s*(sxxha_sxhxh_kf.T.dot(kdk).dot(self.Sxh[t+1]))
            #
            self.Sxh[t][:] = akf.T.dot(self.Sxh[t+1]) - self.Gb[t].T.dot(self.k[t]) \
                + self.s*(sxxh_a_kf.T.dot(mdata.cc).dot(self.Sx[t+1])
                          + sxhxh_a_kf.T.dot(kdk).dot(self.Sxh[t+1]))
            #
            self.Sxx[t][:, :] = data.Lxx + \
                data.Fx.T.dot(self.Sxx[t+1]).dot(data.Fx)\
                + (kf.T.dot(self.Sxhxh[t+1]) + 2. * data.Fx.T.dot(self.Sxxh[t+1])).dot(kf)\
                + self.s*(sxxa_sxxh_kf.T.dot(mdata.cc).dot(sxxa_sxxh_kf)
                          + sxxha_sxhxh_kf.T.dot(kdk).dot(sxxha_sxhxh_kf))
            #
            try:
                self.Sxhxh[t][:, :] = akf.T.dot(sxhxh_a_kf) + (self.Gx[t] + self.Gb[t]).T.\
                    dot(scl.cho_solve(Lb, (self.Gx[t]-self.Gb[t])))\
                    + self.s * (sxxh_a_kf.T.dot(mdata.cc).dot(sxxh_a_kf)
                                + sxhxh_a_kf.T.dot(kdk).dot(sxhxh_a_kf))
            except:
                raise BaseException('backward error')

            #
            self.Sxxh[t][:, :] = sxxha_sxhxh_kf.T.dot(akf) - self.Gx[t].T.dot(self.K[t])\
                + self.s * (sxxa_sxxh_kf.T.dot(mdata.cc).dot(sxxh_a_kf)
                            + sxxha_sxhxh_kf.T.dot(kdk).dot(sxhxh_a_kf))
            # # ensure hessians are symmetric
            self.Sxx[t][:, :] = .5 * (self.Sxx[t] + self.Sxx[t].T)
            self.Sxhxh[t][:, :] = .5 * (self.Sxhxh[t] + self.Sxhxh[t].T)
            self.Sxxh[t][:, :] = .5 * (self.Sxxh[t] + self.Sxxh[t].T)

    def allocateData(self):
        """  Allocate matrix space of Q,V and K.
        Done at init time (redo if problem change).
        """

        self.filterGains = [np.zeros([m.state.ndx, y.ny])for m, y
                            in zip(self.problem.runningModels, self.measurement.measurementModels)]
        self.K = [np.zeros([m.nu, m.state.ndx]) for m in self.problem.runningModels]
        self.k = [np.zeros([m.nu]) for m in self.problem.runningModels]
        #
        self.xs_try = [self.problem.x0] + [np.nan] * self.problem.T
        self.us_try = [np.nan] * self.problem.T
        #
        self.dS = [np.zeros([2*m.state.ndx]) for m in self.models()]
        assert len(self.dS)==self.problem.T+1, "dS wrong dimensions"
        self.Sx = [s[:m.state.ndx] for m, s in zip(self.models(), self.dS)]
        assert len(self.Sx)==self.problem.T+1, "Sx wrong dimensions"
        self.Sxh = [s[m.state.ndx:] for m, s in zip(self.models(), self.dS)]
        assert len(self.Sxh)==self.problem.T+1, "Sxh wrong dimensions"
        self.ddS = [np.zeros([2*m.state.ndx, 2*m.state.ndx]) for m in self.models()]
        assert len(self.ddS)==self.problem.T+1, "ddS wrong dimensions"
        self.Sxx = [S[:m.state.ndx, :m.state.ndx] for m, S in zip(self.models(), self.ddS)]
        assert len(self.Sxx)==self.problem.T+1, "Sxx wrong dimensions"
        self.Sxxh = [S[:m.state.ndx, m.state.ndx:]
                     for m, S in zip(self.models(), self.ddS)]
        assert len(self.Sxxh)==self.problem.T+1, "Sxxh wrong dimensions"
        self.Sxhxh = [S[m.state.ndx:, m.state.ndx:]
                      for m, S in zip(self.models(), self.ddS)]
        assert len(self.Sxhxh)==self.problem.T+1, "Sxhxh wrong dimensions"
        #
        self.H = [np.zeros([m.nu, m.nu]) for m in self.problem.runningModels]
        assert len(self.H)==self.problem.T, "H wrong dimensions"
        self.g = [np.zeros(m.nu) for m in self.problem.runningModels]
        assert len(self.g)==self.problem.T, "g wrong dimensions"
        self.Gx = [np.zeros([m.nu, m.state.ndx]) for m in self.problem.runningModels]
        assert len(self.Gx)==self.problem.T, "Gx wrong dimensions"
        self.Gb = [np.zeros([m.nu, m.state.ndx]) for m in self.problem.runningModels]
        assert len(self.Gb)==self.problem.T, "Gb wrong dimensions"

        self.Q0 = [None for _ in self.models()]
        assert len(self.Q0)==self.problem.T+1, "Q0 wrong dimensions"
        self.cov = [np.zeros([m.state.ndx, m.state.ndx]) for m in self.models()]
        assert len(self.cov)==self.problem.T+1, "cov wrong dimensions"
        # 
        self.fs = [np.zeros(self.problem.runningModels[0].state.ndx)
                     ] + [np.zeros(m.state.ndx) for m in self.problem.runningModels]
        assert len(self.fs)==self.problem.T+1, "gaps wrong dimensions"
        self.cov[0] = self.measurement.measurementModels[0].sn.copy()