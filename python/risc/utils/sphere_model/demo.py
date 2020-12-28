import numpy as np 
import matplotlib.pyplot as plt 
import sys
from os.path import dirname, join, abspath
#
src_path = abspath('../../')
sys.path.append(src_path)

import pinocchio as pin 
import crocoddyl

from utils.sphere_model import sphere_model

if __name__=="__main__": 
    # start by loading sphere model 
    sphereModel = sphere_model.load_sphere_model()
    
    # 
    sphereControl = sphere_model.SE3ControlProblem(sphereModel.model)
    ref = pin.SE3.Random()
    refvec = pin.SE3ToXYZQUAT(ref) 
    shooting_problem = sphereControl.createPositionControlProblem(ref, 1.e-2, 100)
    solver = crocoddyl.SolverDDP(shooting_problem)

    xs = [sphereModel.model.defaultState] *(len(solver.problem.runningModels) + 1)
    us = [np.zeros(6)]*len(solver.problem.runningModels)

    solver.setCallbacks([crocoddyl.CallbackLogger(),
                        crocoddyl.CallbackVerbose(),])

    solver.solve(xs, us, 100, False, 0.1)

    sphere_model.plot_tracking_solution(solver.xs, solver.us, refvec)