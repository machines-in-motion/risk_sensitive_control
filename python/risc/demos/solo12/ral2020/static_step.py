import numpy as np 
import os, sys, time 
src_path = os.path.abspath('../../../src/py_locomotion/')
sys.path.append(src_path)
import matplotlib.pyplot as plt 
import pinocchio as pin 
import crocoddyl 

import robots, locomotion_tools, plotting_tools, measurement 
import risk_sensitive_solver


# some parameters for the gait, solver and simulator 
timeStep=1.e-2 
supportKnots = 10
stepKnots = 40 
footHeight = .1
sensitivity = .9
DISPLAY_DDP_SOLN = False    
DISPLAY_RISK_SOLN = False    
COMPARE_SOLVERS = True 
WHICH_SIMULATION = "ConSim"
BULLET_SIMULATION = False  
CONSIM_SIMULATION = True 
NO_FEEDBACK = False 
# WHICH_SOLVER = "RISK"             
WHICH_SOLVER = "FDDP" 
USE_FDDP = True 
USE_RS = True  
PLOT_PLANNER_REF = False  

ADD_DISTURBANCE = False     
DISPLAY  = [True, True, False]  # gepetto display for planner, solver and simulation

disturbance_position = [.4, .147, .015+.015] #[.3, .147, .015+.015]
disturbance_position2 = [.5, -.1225, .0075]
# [.19, .147, .0075]
# disturbance_orientation = p.getQuaternionFromEuler([0,0,0])
disturbance_size = [.1, .1, .03]
color_map = "BrBG"

noise_models = ["Uniform", "SwingJoints", "Unconstrained", "Contact"]

contact_names = ['FL_ANKLE', 'FR_ANKLE', 'HL_ANKLE', 'HR_ANKLE']
solo_path = os.path.abspath('../../../../robot_properties_solo')
plan_path = os.path.abspath("../planner/sec_static_ref")

if __name__=="__main__":

    solo12 = robots.load_solo12_pinocchio(solo_path)
    solo12_plots = plotting_tools.RobotPlottingTools(solo12, contact_names)

    x0 = solo12.defaultState

    """ read kino_dynamic_planner """

    qref, dqref, fref, vref, contact_positions_ref, \
    contact_status_ref, com_ref = locomotion_tools.parse_kindynamic_plan_slo12(plan_path, solo_path)

    t_array = timeStep * np.arange(qref.shape[0])
    label_direction = [' x ', ' y ', ' z ']

    """ Plot KinoDynamicPlanner Reference Trajectories """
    if PLOT_PLANNER_REF:
        # contact forces 
        fig, ax = plt.subplots(3,1)
        for i in range(3):
            for j in range(4):
                ax[i].plot(t_array, fref[:,3*j+i], label=contact_names[j][:2]+label_direction[i]+'force')
            ax[i].legend()
            ax[i].grid()
        fig.canvas.set_window_title('contact forces')
        # contact positions
        fig, ax = plt.subplots(3,1)
        for i in range(3):
            for j in range(4):
                ax[i].plot(t_array, contact_positions_ref[:,3*j+i], label=contact_names[j][:2]+label_direction[i]+'pos')
            ax[i].legend()
            ax[i].grid()
        fig.canvas.set_window_title('contact positions')
        # contact velocities
        fig, ax = plt.subplots(3,1)
        for i in range(3):
            for j in range(4):
                ax[i].plot(t_array, vref[:,3*j+i], label=contact_names[j][:2]+label_direction[i]+'vel')
            ax[i].legend()
            ax[i].grid()
        fig.canvas.set_window_title('contact velocities')
        # com ref 
        plt.figure()
        for i in range(3):
            plt.plot(t_array, com_ref[:,i], label='com'+label_direction[i])
        plt.grid()
        plt.legend()

    if any(DISPLAY):
        try:
            solo12.initViewer(loadModel=True)
            cameraTF = [3., 3.68, 0.84, 0.2, 0.62, 0.72, 0.22]
            window_id = solo12.viz.viewer.gui.getWindowID("python-pinocchio")
            solo12.viewer.gui.setCameraTransform(window_id, cameraTF)
            backgroundColor = [1., 1., 1., 1.]
            floorColor = [0.7, 0.7, 0.7, 1.]
            #   
            
            solo12.viz.viewer.gui.setBackgroundColor1(window_id, backgroundColor)
            solo12.viz.viewer.gui.setBackgroundColor2(window_id, backgroundColor)
            solo12.display(x0[:solo12.nq])
        except:
            raise BaseException('Gepetto viewer not initialized ! ')

    if DISPLAY[0]:
        print("DISPLAYING KINO-DYNAMIC SOLUTION")
        for qi in qref:
            solo12.display(qi)
            time.sleep(timeStep)

    solo12_gaits = locomotion_tools.QuadrupedGaits(solo12, *contact_names)
    x0 = np.hstack([qref[0], dqref[0]])
    x0[2] -= solo12_gaits.ankle_offset 

    loco3dModels, measurementModels = solo12_gaits.createProblemStateTracking(x0, timeStep, 
                    contact_status_ref, qref, dqref, contact_positions_ref,vref)

    # loco3dModels, measurementModels = solo12_gaits.createProblemFromKinoDynPlanner(x0, timeStep, 
    # contact_status_ref, com_ref, contact_positions_ref, vref)
    

    kinoOptProblem = crocoddyl.ShootingProblem(x0, loco3dModels, loco3dModels[-1])

    """ FDDP Solver """
    fddp =  crocoddyl.SolverFDDP(kinoOptProblem)
    fddp.setCallbacks(
            [crocoddyl.CallbackLogger(),
            crocoddyl.CallbackVerbose()])
    # set initial guess and solve using FDDP
    xs = [solo12.defaultState] * (fddp.problem.T + 1)
    us = [m.quasiStatic(d, solo12.defaultState) for m, d in list(zip(fddp.problem.runningModels, fddp.problem.runningDatas))]
    fddp.solve(xs, us, 1000, False, 0.1)
    
    
    
    if DISPLAY[1]:
        print("displaying FDDP Solution")
        for xi in fddp.xs:
            solo12.display(xi[:solo12.nq])
            time.sleep(1.e-2)

