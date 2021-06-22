""" includes a demo to setup cliff problem with risk sensitive stuff """

import numpy as np 
from cliff_diff_action import DifferentialActionModelCliff
from os.path import dirname, join, abspath

# import measurement 
import crocoddyl 
import matplotlib.pyplot as plt 
# add package path manually
import os
import sys
#
src_path = abspath('../../')
sys.path.append(src_path)

from solvers import risc 
from utils import measurement 


MAXITER = 1000
CALLBACKS = True
SENSITIVITY  = .1
dt = 0.01 
T = 300 
x0 = np.zeros(4) 


def ddpCreateProblem(model):
    cliff_diff_running =  DifferentialActionModelCliff()
    cliff_diff_terminal = DifferentialActionModelCliff(isTerminal=True)
    cliff_running = crocoddyl.IntegratedActionModelEuler(cliff_diff_running, dt) 
    cliff_terminal = crocoddyl.IntegratedActionModelEuler(cliff_diff_terminal, dt) 
    xs = [x0]*(T+1)
    us = [np.zeros(2)]*T
    problem = crocoddyl.ShootingProblem(x0, [cliff_running]*T, cliff_terminal)
    return xs, us, problem


def riskCreateProblem(model):
    ### also add measurement models compared to ddp 
    
    cliff_diff_running =  DifferentialActionModelCliff()
    cliff_diff_terminal = DifferentialActionModelCliff(isTerminal=True)
    cliff_running = crocoddyl.IntegratedActionModelEuler(cliff_diff_running, dt) 
    cliff_terminal = crocoddyl.IntegratedActionModelEuler(cliff_diff_terminal, dt) 
    xs = [x0]*(T+1)
    us = [np.array([0., 9.81])]*T
    runningModels = [cliff_running]*T
    terminalModel = cliff_terminal
    runningMeasurements = []

    for t, process_model in enumerate(runningModels):
        state_diffusion = dt * np.eye(process_model.state.ndx)
        state_diffusion[1,1] = 2*dt 
        state_noise =  np.eye(process_model.state.ndx)
        measurement_diffusion = np.eye(process_model.state.ndx)
        measurement_noise = 1.e-4 * np.eye(process_model.state.ndx) 
        measurementMod = measurement.MeasurementModelFullState(process_model,state_diffusion, 
                    state_noise, measurement_diffusion, measurement_noise)
        runningMeasurements += [measurementMod]
    return xs, us, runningModels, terminalModel, runningMeasurements





if __name__ == "__main__": 
    # create the problem 

    ddp_xs, ddp_us, ddp_problem = ddpCreateProblem(crocoddyl.ActionModelLQR)
    ddp = crocoddyl.SolverDDP(ddp_problem)
    if CALLBACKS:
        ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
    
    ddp.solve(ddp_xs, ddp_us, MAXITER)

    risk_xs, risk_us, runningModels, terminalModel, runningMeasurements = riskCreateProblem(crocoddyl.ActionModelLQR)

    riskProblem = crocoddyl.ShootingProblem(risk_xs[0], runningModels, terminalModel)
    measurementModels = measurement.MeasurementModels(runningModels, runningMeasurements)
    print("measurement models initialized successfully ")

    riskSolver = risc.RiskSensitiveSolver(riskProblem, measurementModels, SENSITIVITY)
    riskSolver.callback = [crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()]
    print("risk solver initialized successfully ")


    riskSolver.solve(MAXITER, risk_xs, risk_us, False)


    time_array = dt*np.arange(np.array(ddp.xs).shape[0])

    plt.figure("State_Trajectory")
    plt.plot(np.array(ddp.xs)[:,0], np.array(ddp.xs)[:,1])
    plt.plot(np.array(riskSolver.xs)[:,0], np.array(riskSolver.xs)[:,1])

    plt.figure("X controls")
    plt.plot(time_array[:-1], np.array(ddp.us)[:,0])
    plt.plot(time_array[:-1], np.array(riskSolver.us)[:,0])
    
    plt.figure("Y controls")
    plt.plot(time_array[:-1], np.array(ddp.us)[:,1])
    plt.plot(time_array[:-1], np.array(riskSolver.us)[:,1])

    plt.figure("X feedback")
    plt.plot(time_array[:-1], -np.array(ddp.K)[:,0,0])
    plt.plot(time_array[:-1], np.array(riskSolver.K)[:,0,0])

    plt.figure("Y feedback")
    plt.plot(time_array[:-1], -np.array(ddp.K)[:,1,1])
    plt.plot(time_array[:-1], np.array(riskSolver.K)[:,1,1])
    

    # plt.figure("Control_Solution")
    # for i in range(NU):
    #     plt.plot(np.arange(N), np.array(ddp.us)[:,i])
    #     plt.plot(np.arange(N), np.array(riskSolver.us)[:,i])

    plt.show()
