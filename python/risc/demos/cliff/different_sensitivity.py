""" Run risk process solver with a range of sensitivity paramters from -10 to + 10
given a fixed process noise sequence  """

LINE_WIDTH = 100 
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
SENSITIVITY  = 5. 
dt = 0.01 
T = 300 
x0 = np.zeros(4) 

sigmas = np.arange(-10,10)

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
        # state_diffusion[1,1] = 2*dt 
        # state_diffusion[3,3] = 5*dt 
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

    solvers =[ddp]
    solver_names = ["ddp"]
    

    for i, sig in enumerate(sigmas):
        print("Running risk sensitive with sensitivity %s".center(LINE_WIDTH,'#')%(sig))
        risk_xs, risk_us, runningModels, terminalModel, runningMeasurements = riskCreateProblem(crocoddyl.ActionModelLQR)
        riskProblem = crocoddyl.ShootingProblem(risk_xs[0], runningModels, terminalModel)
        measurementModels = measurement.MeasurementModels(runningModels, runningMeasurements)
        # print("measurement models initialized successfully ")

        riskSolver = risc.RiskSensitiveSolver(riskProblem, measurementModels, sig)
        riskSolver.callback = [crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()]
        # print("risk solver initialized successfully ")

        riskSolver.solve(MAXITER, risk_xs, risk_us, False)
        solvers += [riskSolver]
        solver_names += ["risk %s"%sig]

    time_array = dt*np.arange(np.array(ddp.xs).shape[0])

    plt.figure("State_Trajectory")
    for i, solver in enumerate(solvers): 
        plt.plot(np.array(solver.xs)[:,0], np.array(solver.xs)[:,1], label=solver_names[i])
    plt.legend()
    plt.xlabel('X position')
    plt.ylabel('Z position')
    plt.axes().set_aspect('equal', 'datalim')

    # plt.figure("X controls")
    # plt.plot(time_array[:-1], np.array(ddp.us)[:,0])
    # plt.plot(time_array[:-1], np.array(riskSolver.us)[:,0])
    
    # plt.figure("Y controls")
    # plt.plot(time_array[:-1], np.array(ddp.us)[:,1])
    # plt.plot(time_array[:-1], np.array(riskSolver.us)[:,1])

    # plt.figure("Kp Y")
    # plt.plot(time_array[:-1], -np.array(ddp.K)[:,1,1])
    # plt.plot(time_array[:-1], np.array(riskSolver.K)[:,1,1])

    # plt.figure("Kd Y")
    # plt.plot(time_array[:-1], -np.array(ddp.K)[:,1,3])
    # plt.plot(time_array[:-1], np.array(riskSolver.K)[:,1,3])
    

    # plt.figure("Control_Solution")
    # for i in range(NU):
    #     plt.plot(np.arange(N), np.array(ddp.us)[:,i])
    #     plt.plot(np.arange(N), np.array(riskSolver.us)[:,i])

    plt.show()
