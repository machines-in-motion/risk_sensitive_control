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



NX = 37  # state dimension 
NU = 12  # control dimension 
N = 100  # number of nodes

MAXITER = 10
CALLBACKS = True


def ddpCreateProblem(model):
    x0 = np.zeros(NX)
    runningModels = [model(NX, NU)] * N
    terminalModel = model(NX, NU)
    xs = [x0] * (N + 1)
    us = [np.zeros(NU)] * N

    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)
    return xs, us, problem


def riskCreateProblem(model):
    ### also add measurement models compared to ddp 
    pass 




if __name__ == "__main__": 
    # create the problem 

    xs, us, problem = ddpCreateProblem(crocoddyl.ActionModelLQR)
    ddp = crocoddyl.SolverDDP(problem)
    if CALLBACKS:
        ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
    
    ddp.solve(xs, us, MAXITER)



