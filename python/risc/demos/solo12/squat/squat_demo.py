""" Solo squatting demo using both ddp and risk sensitive control """ 


import numpy as np 
import os, sys, time 
src_path = os.path.abspath('../../../') # append library directory without packaging 
sys.path.append(src_path)
import matplotlib.pyplot as plt 
import pinocchio as pin 
import crocoddyl 
from robot_properties_solo.config import Solo12Config
from utils import plotting_tools, measurement, simple_simulator 
from solvers import risc
import seaborn as sns 
import squatting_problem 

timeStep=1.e-2 
sensitivity = .1
solo12_config = Solo12Config() 
horizon = 300 
contact_names = ["FL_ANKLE", "FR_ANKLE", "HL_ANKLE", "HR_ANKLE"]
# 
LINE_WIDTH = 100 # printing purposes 
MAXITER = 100 


# noise_models = ["Uniform", "SwingJoints","Unconstrained", "Contact"]
noise_models = ["Uniform"]

if __name__ =="__main__":
    # load solo 
    solo12 = solo12_config.pin_robot
    contact_names = solo12_config.end_effector_names
    # plotting tools 
    solo12_plots = plotting_tools.RobotPlottingTools(solo12, contact_names)
    x0 = np.hstack([solo12_config.q0, solo12_config.v0])
    solo12.model.referenceConfigurations["standing"] = solo12_config.q0.copy()
    solo12.defaultState = x0.copy()
    time_array = timeStep*np.arange(horizon)
    # ocp setup 
    squatting = squatting_problem.QuadrupedSquatting(solo12, *contact_names)
    squatting.WHICH_MEASUREMENT = "Uniform"
    loco3dModels, runningMeasurements = squatting.createBalanceProblem(x0, timeStep, 
                    horizon)
    print("Optimal Control Problem Constructed".center(LINE_WIDTH,'-'))

    problem = crocoddyl.ShootingProblem(x0, loco3dModels[:-1], loco3dModels[-1])

    """ FDDP Solver """
    print("Solving FDDP".center(LINE_WIDTH,'-'))
    fddp =  crocoddyl.SolverFDDP(problem)
    fddp.setCallbacks(
            [crocoddyl.CallbackLogger(),
            crocoddyl.CallbackVerbose()])
    # set initial guess and solve using FDDP
    xs = [solo12.defaultState] * (fddp.problem.T + 1)
    us = [m.quasiStatic(d, solo12.defaultState) for m, d in list(zip(fddp.problem.runningModels, fddp.problem.runningDatas))]
    fddp.solve(xs, us, 1000, False, 0.1)
    print("Solving FDDP Completed".center(LINE_WIDTH,'-'))

    solvers = [fddp]
    solver_names = ["fddp"]

    print("Setting up Risk Sensitive".center(LINE_WIDTH,'-'))

    riskProblem = crocoddyl.ShootingProblem(x0, loco3dModels[:-1], loco3dModels[-1])
    measurementModels = measurement.MeasurementModels(loco3dModels, runningMeasurements)
    print("measurement models initialized successfully ")

    processRiskSolver = risc.RiskSensitiveSolver(riskProblem, measurementModels, sensitivity)
    processRiskSolver.callback = [crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()]
    print("risk solver initialized successfully ")

    risk_xs = [xi for xi in fddp.xs]
    risk_us = [ui for ui in fddp.us]

    processRiskSolver.solve(MAXITER, risk_xs, risk_us, False)
    print("risk solver finished successfully ")

    solvers += [processRiskSolver]
    solver_names += ["process_risk_uniform"]


    measurementRiskSolver = risc.RiskSensitiveSolver(riskProblem, measurementModels, sensitivity)
    measurementRiskSolver.callback = [crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()]
    print("risk solver initialized successfully ")

    risk_xs = [xi for xi in fddp.xs]
    risk_us = [ui for ui in fddp.us]

    measurementRiskSolver.solve(MAXITER, risk_xs, risk_us, True)
    print("risk solver finished successfully ")

    solvers += [measurementRiskSolver]
    solver_names += ["measurement_risk_uniform"]


    plt.rc('legend', fontsize=30)    # legend fontsize
    fig, ax = plt.subplots(3, 6,figsize=(30,15))

    for i in range(3):
        for j in range(6): 
            for k, solver in enumerate(solvers): 
                if "ddp" in solver_names[k]:
                    ax[i,j].plot(time_array[:-1], -np.array(solver.K)[:,i,j], 'k', linewidth=5., label=solver_names[k])
                else:
                    ax[i,j].plot(time_array[:-1], np.array(solver.K)[:,i,j],linewidth=2., label=solver_names[k])
    ax[0,5].legend(loc="center left", bbox_to_anchor=(1., -.1), ncol= 1)

    fig, ax = plt.subplots(3, 12,figsize=(30,15))
    for i in range(3):
        for j in range(6,18): 
            for k, solver in enumerate(solvers): 
                if "ddp" in solver_names[k]:
                    ax[i,j-6].plot(time_array[:-1], -np.array(solver.K)[:,i,j], 'k', linewidth=5., label=solver_names[k])
                else:
                    ax[i,j-6].plot(time_array[:-1], np.array(solver.K)[:,i,j],linewidth=2., label=solver_names[k])
    ax[0,11].legend(loc="center left", bbox_to_anchor=(1., -.1), ncol= 1)

    fig, ax = plt.subplots(3, 6,figsize=(30,15))
    for i in range(3):
        for j in range(18,24): 
            for k, solver in enumerate(solvers): 
                if "ddp" in solver_names[k]:
                    ax[i,j-18].plot(time_array[:-1], -np.array(solver.K)[:,i,j], 'k', linewidth=5., label=solver_names[k])
                else:
                    ax[i,j-18].plot(time_array[:-1], np.array(solver.K)[:,i,j],linewidth=2., label=solver_names[k])
    ax[0,5].legend(loc="center left", bbox_to_anchor=(1., -.1), ncol= 1)

    fig, ax = plt.subplots(3, 12,figsize=(30,15))
    for i in range(3):
        for j in range(24,36): 
            for k, solver in enumerate(solvers): 
                if "ddp" in solver_names[k]:
                    ax[i,j-24].plot(time_array[:-1], -np.array(solver.K)[:,i,j], 'k', linewidth=5., label=solver_names[k])
                else:
                    ax[i,j-24].plot(time_array[:-1], np.array(solver.K)[:,i,j],linewidth=2., label=solver_names[k])
    ax[0,11].legend(loc="center left", bbox_to_anchor=(1., -.1), ncol= 1)

    plt.show()