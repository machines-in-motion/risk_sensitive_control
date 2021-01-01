import numpy as np 
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
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

from solvers import measurement_risc, risk_sensitive_solver 
from utils import measurement 

NX = 37  # state dimension 
NU = 12  # control dimension 
N = 100  # number of nodes

MAXITER = 10
CALLBACKS = True
SENSITIVITY  = 1. 


def ddpCreateProblem(model):
    x0 = 10. * np.ones(NX)
    runningModels = [model(NX, NU)] * N
    terminalModel = model(NX, NU)
    xs = [x0] * (N + 1)
    us = [np.zeros(NU)] * N

    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)
    return xs, us, problem


def riskCreateProblem(model):
    ### also add measurement models compared to ddp 
    
    x0 = 10. * np.ones(NX)
    xs = [x0] * (N + 1)
    us = [np.zeros(NU)] * N

    runningModels = [model(NX, NU)] * N
    terminalModel = model(NX, NU)

    runningMeasurements = []

    for t, process_model in enumerate(runningModels):
        state_diffusion = .1 * np.eye(process_model.state.ndx)
        state_noise = 1.e-7 * np.eye(process_model.state.ndx)
        measurement_diffusion = .1 * np.eye(process_model.state.ndx)
        measurement_noise = 1.e-2 * np.eye(process_model.state.ndx) 
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


    # riskSolver = measurement_risc.MeasurementRiskSensitiveSolver(riskProblem, measurementModels, SENSITIVITY)

    riskSolver = risk_sensitive_solver.RiskSensitiveSolver(riskProblem, measurementModels, SENSITIVITY)
    riskSolver.callback = [crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()]
    print("risk solver initialized successfully ")


    riskSolver.solve(MAXITER, risk_xs, risk_us, True)

    plt.figure("State_Solution")
    for i in range(NX):
        plt.plot(np.arange(N+1), np.array(ddp.xs)[:,i])
        plt.plot(np.arange(N+1), np.array(riskSolver.xs)[:,i])


    plt.figure("Control_Solution")
    for i in range(NU):
        plt.plot(np.arange(N), np.array(ddp.us)[:,i])
        plt.plot(np.arange(N), np.array(riskSolver.us)[:,i])

    plt.show()


