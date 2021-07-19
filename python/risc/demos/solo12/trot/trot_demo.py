""" Solo trotting demo using both ddp and risk sensitive control """ 


import numpy as np 
import os, sys, time 
src_path = os.path.abspath('../../../') # append library directory without packaging 
sys.path.append(src_path)
import matplotlib.pyplot as plt 
import pinocchio as pin 
import crocoddyl 
from robot_properties_solo.config import Solo12Config
from utils import plotting_tools, measurement, simple_simulator , locomotion_tools
from solvers import risc
import seaborn as sns 


timeStep=1.e-2 
sensitivity = .1 
solo12_config = Solo12Config() 
horizon = 300 
contact_names = ["FL_ANKLE", "FR_ANKLE", "HL_ANKLE", "HR_ANKLE"]
# 
LINE_WIDTH = 100 # printing purposes 
MAXITER = 100 
WITH_SIMULATION = False   
PLOT_FEEDBACK = False 

plan_path = '../planner/trot_ref/'

# noise_models = ["Uniform", "SwingJoints","Unconstrained", "Contact"]
noise_model = "Uniform"

# plotting flags mainly for debugging purposes 
PLOT_PLANNER_REF = False    

SAVE_SOLUTION = False  


if __name__ =="__main__":
    # load solo 
    solo12 = solo12_config.pin_robot
    # plotting tools 
    solo12_plots = plotting_tools.RobotPlottingTools(solo12, contact_names)
    # reading kinodynamic plan
    qref, dqref, fref, vref, contact_positions_ref, \
    contact_status_ref, com_ref = locomotion_tools.parse_kindynamic_plan_slo12(plan_path, solo12, contact_names)
     # initialization states 
    solo12.model.referenceConfigurations["standing"] = qref[0]
    x0 = np.concatenate([qref[0], np.zeros(solo12.model.nv)])
    solo12.defaultState = x0.copy()
    print(" Kinodynamic Plan Loaded Successfully ".center(LINE_WIDTH,'-'))

    time_array = timeStep * np.arange(qref.shape[0]) # assumes all kinodyn trajectories are same dimension

    label_direction = [' x ', ' y ', ' z ']
    """ Plot KinoDynamicPlanner Reference Trajectories """
    if PLOT_PLANNER_REF:
        # contact forces 
        fig, ax = plt.subplots(3,1)
        for i in range(3):
            for j in range(4):
                ax[i].plot(time_array, fref[:,3*j+i], label=contact_names[j][:2]+label_direction[i]+'force')
            ax[i].legend()
            ax[i].grid()
        fig.canvas.set_window_title('contact forces')
        # contact positions
        fig, ax = plt.subplots(3,1)
        for i in range(3):
            for j in range(4):
                ax[i].plot(time_array, contact_positions_ref[:,3*j+i], label=contact_names[j][:2]+label_direction[i]+'pos')
            ax[i].legend()
            ax[i].grid()
        fig.canvas.set_window_title('contact positions')
        # contact velocities
        fig, ax = plt.subplots(3,1)
        for i in range(3):
            for j in range(4):
                ax[i].plot(time_array, vref[:,3*j+i], label=contact_names[j][:2]+label_direction[i]+'vel')
            ax[i].legend()
            ax[i].grid()
        fig.canvas.set_window_title('contact velocities')
        # com ref 
        plt.figure()
        for i in range(3):
            plt.plot(time_array, com_ref[:,i], label='com'+label_direction[i])
        plt.grid()
        plt.legend()
        plt.title('CoM Ref trajectory')

        orientation_names = ['qx', 'qy', 'qz', 'qw']
        plt.figure()
        for i in range(4):
            plt.plot(time_array, qref[:,i+3], label=orientation_names[i])
        plt.grid()
        plt.legend()
        plt.title('base Orientation ')

    """ Now back to running stuff """

    solo12_gaits = locomotion_tools.QuadrupedGaits(solo12, *contact_names)
    solo12_gaits.WHICH_MEASUREMENT = noise_model 

    loco3dModels, runningMeasurements = solo12_gaits.createProblemStateTracking(x0, timeStep, 
        contact_status_ref, qref, dqref, contact_positions_ref, vref)
    print(" OCP models generated successfully ".center(LINE_WIDTH,'-'))

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

    """ Risk sensitive with process noise only """
    print(" Setting up Risk Sensitive with Process Noise ".center(LINE_WIDTH,'-'))

    riskProblem = crocoddyl.ShootingProblem(x0, loco3dModels[:-1], loco3dModels[-1])
    measurementModels = measurement.MeasurementModels(loco3dModels, runningMeasurements)

    processRiskSolver = risc.RiskSensitiveSolver(riskProblem, measurementModels, sensitivity)
    processRiskSolver.callback = [crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()]

    risk_xs = [xi for xi in fddp.xs]
    risk_us = [ui for ui in fddp.us]

    processRiskSolver.solve(MAXITER, risk_xs, risk_us, False)

    solvers += [processRiskSolver]
    solver_names += ["process_risk_uniform"]

    print("Solving Process Risk Sensitive Completed".center(LINE_WIDTH,'-'))


    """ Risk sensitive with both process and measurement noise """
    print(" Setting up Risk Sensitive with Measurement Noise ".center(LINE_WIDTH,'-'))

    measurementRiskSolver = risc.RiskSensitiveSolver(riskProblem, measurementModels, sensitivity)
    measurementRiskSolver.callback = [crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()]

    risk_xs = [xi for xi in fddp.xs]
    risk_us = [ui for ui in fddp.us]

    measurementRiskSolver.solve(MAXITER, risk_xs, risk_us, True)

    solvers += [measurementRiskSolver]
    solver_names += ["measurement_risk_uniform"]

    print("Solving Measurement Risk Sensitive Completed".center(LINE_WIDTH,'-'))


    if PLOT_FEEDBACK:
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

    if WITH_SIMULATION:
        print(" Running Simulations ".center(LINE_WIDTH,'#'))
        contact_config_path = '../ral2020/linear_contact.yaml'
        sim = simple_simulator.SimpleSimulator(solo12, contact_config_path)

        states = []
        forces = []
        controls = []
        errors = []
        for i, soli in enumerate(solvers): 
            x0 = np.resize(soli.xs[0].copy(), soli.xs[0].shape[0])
            xt = [x0.copy()]
            sim.reset_state(x0.copy())
            ft = [sim.read_contact_force()]
            ut = []
            dx = []
            for t, ui in enumerate(soli.us):
                for d in range(10):
                        # diff(xact, xref) = xref [-] xact --> on manifold 
                    xref = solo12_gaits.interpolate_state(soli.xs[t], soli.xs[t+1], .1*d)
                    xact = sim.read_state()
                    diff = solo12_gaits.state.diff(xact, xref)
                    if 'ddp' in solver_names[i]:
                        u = np.resize(ui + soli.K[t].dot(diff), sim.m)
                    else:
                        u = np.resize(ui - soli.K[t].dot(diff), sim.m)
                    ut += [u]
                    sim.integrate_step(u)
                xt += [sim.read_state()]
                ft += [sim.read_contact_force()]
                dx += [diff]
    #             if t == 130:
    #                 break
    #             # display in gepetto 
    #             # solo12.display(xt[-1][:solo12.nq,None])
    #             # time.sleep(1.e-2)
            # print("Displaying Solo12 Simulation for %s"%solver_names[i])
            # for t, xi in enumerate(xt):
            #     solo12.display(xi[:solo12.nq])
            #     time.sleep(1.e-2) 

            states += [xt]
            forces += [ft]
            controls += [ut]
            errors += [dx]
        base_tracking_fig = solo12_plots.compare_base_tracking(states, fddp.xs, names=solver_names)
        contact_height_tracking_fig = solo12_plots.compare_frame_height(contact_names, states, fddp.xs, names=solver_names ,setlim=False)
        contact_force_fig = solo12_plots.compare_simulation_froces(fddp, forces, dt=1.e-2, names=solver_names)


    if SAVE_SOLUTION: 
        for i, soli in enumerate(solvers):
            np.save(solver_names[i]+'_x_ref', np.array(soli.xs)) 
            np.save(solver_names[i]+'_u_ref', np.array(soli.us)) 
            if "ddp" in solver_names[i]:
                np.save(solver_names[i]+'_K_ref', -np.array(soli.K))  
            else: 
                np.save(solver_names[i]+'_K_ref', np.array(soli.K))   


    plt.show()