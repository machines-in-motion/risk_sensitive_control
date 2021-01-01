import numpy as np 
import os, sys, time 
src_path = os.path.abspath('../../../src/py_locomotion/')
sys.path.append(src_path)
import matplotlib.pyplot as plt 
import pinocchio as pin 
import crocoddyl 

import robots, locomotion_tools, plotting_tools, measurement 
import risk_sensitive_solver

import seaborn as sns 



EXPERIMENT_NAME="Solo12_single_step"
SAVE_FIGURES =True   
SMALL_SIZE = 8
MEDIUM_SIZE = 50
BIGGER_SIZE = 22

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title


# some parameters for the gait, solver and simulator 
timeStep=1.e-2 
supportKnots = 10
stepKnots = 40 
footHeight = .1
sensitivity = -7.5
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

ADD_DISTURBANCE = True     
disturbance_position = [.3, .147, .015+.015] #[.3, .147, .015+.015]
disturbance_position2 = [.5, -.1225, .0075]
disturbance_size = [.1, .1, .03]
color_map = "BrBG"
# experiment_path = 'contact_consistent_noise'
# experiment_path = 'unconstrained_contact_noise'
experiment_path = 'joint_noise'
noise_models = ["Uniform", "SwingJoints","Unconstrained", "Contact"]
solo_path = os.path.abspath('../../../../robot_properties_solo')
contact_names = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']


class MeasurementModelsClass(object):
    def __init__(self, running_models, measurement_models):
        self.rmodels = running_models
        self.measurementModels = measurement_models
        self.runningDatas = []

        for t, mModel in enumerate(self.measurementModels):
            self.runningDatas += [mModel.createData()]
            


if __name__ == "__main__":

    solo12 = robots.load_solo12_pinocchio(solo_path)

    solo12_plots = plotting_tools.RobotPlottingTools(solo12, contact_names)

    x0 = solo12.defaultState

    if WHICH_SIMULATION=="ConSim":
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




    solo12_gaits = locomotion_tools.QuadrupedGaits(solo12, *contact_names)

    loco3dModels, measurementModels = solo12_gaits.createOneFootLiftingProblem(x0, timeStep, 
                                                                            supportKnots, stepKnots, footHeight)



    oneFootLiftingProblem = crocoddyl.ShootingProblem(x0, loco3dModels, loco3dModels[-1])
    """ FDDP Solver """
    print("Solving FDDP")
    fddp =  crocoddyl.SolverFDDP(oneFootLiftingProblem)
    fddp.setCallbacks(
        [crocoddyl.CallbackLogger(),
        crocoddyl.CallbackVerbose()])
    # set initial guess and solve using FDDP
    xs = [solo12.defaultState] * (fddp.problem.T + 1)
    us = [m.quasiStatic(d, solo12.defaultState) for m, d in list(zip(fddp.problem.runningModels, fddp.problem.runningDatas))]
    fddp.solve(xs, us, 1000, False, 0.1)
    print("Solving FDDP Completed")

    """ risk sensitive  """
    risk_solvers_soln = []
    risk_names_dff = ["Risk_"+ n for n in noise_models]
    for i in range(len(noise_models)):
        print("Solving "+ risk_names_dff[i])
        solo12_gaits = locomotion_tools.QuadrupedGaits(solo12, *contact_names)
        solo12_gaits.WHICH_MEASUREMENT = noise_models[i]
        loco3dModels, measurementModels = solo12_gaits.createOneFootLiftingProblem(x0, timeStep, 
                                                                            supportKnots, stepKnots, footHeight)
        oneFootLiftingProblem = crocoddyl.ShootingProblem(x0, loco3dModels, loco3dModels[-1])
        measurementModel = MeasurementModelsClass(oneFootLiftingProblem.runningModels, measurementModels)
        risk_solvers_soln += [risk_sensitive_solver.RiskSensitiveSolver(oneFootLiftingProblem, measurementModel, 
                                                                        sensitivity)]
        risk_solvers_soln[-1].callback = [crocoddyl.CallbackLogger(),
                                crocoddyl.CallbackVerbose()]
        # 
        xs_risk = [xi.copy() for xi in fddp.xs]
        us_risk = [ui.copy() for ui in fddp.us]
        # 
        _, _, converged = risk_solvers_soln[-1].solve(100, xs_risk, us_risk, True)
        print("Solving "+ risk_names_dff[i]+ " Completed")

    solvers = [fddp]+ risk_solvers_soln
    solver_names = ['DDP'] + ["Risk_"+ n for n in noise_models]
    if WHICH_SIMULATION == "ConSim":
        contact_config_path = '../../../models/config/linear_contact.yaml'
        sim = robots.PinocchioSimulator(solo12, contact_config_path)
        # simulator runs at 1 ms while the trajectory is optimized at 10 ms 
        if ADD_DISTURBANCE:
            sim.add_box("box01", np.array(disturbance_position), np.array(disturbance_size), solo12) 
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
                    u = np.resize(ui + soli.K[t].dot(diff), sim.m)
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
            print("Displaying Solo12 Simulation for %s"%solver_names[i])
            for t, xi in enumerate(xt):
                solo12.display(xi[:solo12.nq])
                time.sleep(1.e-2) 

            states += [xt]
            forces += [ft]
            controls += [ut]
            errors += [dx]

    # base_tracking_fig = solo12_plots.compare_base_tracking(states, fddp.xs)
    # contact_height_tracking_fig = solo12_plots.compare_frame_height(contact_names, states, fddp.xs, names=solver_names ,setlim=False)
    # contact_force_fig = solo12_plots.compare_simulation_froces(fddp, forces, dt=1.e-2, names=solver_names)

    # if SAVE_FIGURES:
    #     # base_tracking_fig[0].savefig(EXPERIMENT_NAME+"_base_tracking.pdf")
    #     contact_height_tracking_fig[0].savefig(EXPERIMENT_NAME+"_contact_height_tracking.pdf")
    #     contact_directions = ["_fx.pdf", "_fy.pdf", "_fz.pdf"]
    #     for i,d in enumerate(contact_directions):
    #         contact_force_fig[i].savefig(EXPERIMENT_NAME+ d)



    # FEEDBACKFIGSIZE=(10,10)

    # N = len(errors[0])
    # t_array = 1.e-2 * np.arange(N)
    # colors = ['b', 'r--', 'c:', 'm-.', 'g-']
    # # colors = ['b', 'r', 'g', 'm-.', 'k:']

    # dq = np.zeros([N,len(solver_names)])
    # dv = np.zeros([N,len(solver_names)])
    # kp = np.zeros([N,len(solver_names)])
    # kd = np.zeros([N,len(solver_names)])
    # for i, ithSolverName in enumerate(solver_names):
    #     for t in range(N):
    #         dq[t,i] = np.linalg.norm(errors[i][t][:solo12.nv])
    #         dv[t,i] = np.linalg.norm(errors[i][t][solo12.nv:])
    #         kp[t,i] = np.linalg.norm(solvers[i].K[t][:,:solo12.nv])
    #         kd[t,i] = np.linalg.norm(solvers[i].K[t][:,solo12.nv:])


     
    # plt.figure('configuration error', figsize=FEEDBACKFIGSIZE)
    # for i, ithSolverName in enumerate(solver_names):
    #     plt.plot(t_array, dq[:,i], colors[i], linewidth=4., label=ithSolverName)
    # plt.ylim(-.01, 1.5)
    # plt.legend(loc='upper left')
    # plt.xlabel("time [s]")
    # plt.ylabel("$|\delta q|$")

    # plt.tight_layout()
    # if SAVE_FIGURES:
    #     plt.savefig(EXPERIMENT_NAME + "_qerror.pdf")

    # plt.figure('Kp ddp', figsize=FEEDBACKFIGSIZE)
    # for i, ithSolverName in enumerate(solver_names):
    #     plt.plot(t_array, kp[:,i], colors[i], linewidth=4., label=ithSolverName)
    # plt.xlabel("time [s]")
    # plt.ylabel("$| K_p |$")
    
    # # plt.legend(loc='upper left')
    # plt.tight_layout()
    # if SAVE_FIGURES:
    #     plt.savefig(EXPERIMENT_NAME + "_Kp.pdf")

    # plt.figure('velocity error', figsize=FEEDBACKFIGSIZE)
    # for i, ithSolverName in enumerate(solver_names):
    #     plt.plot(t_array, dv[:,i], colors[i], linewidth=4., label=ithSolverName)
    # plt.ylim(-.1, 30.)
    # # plt.legend(loc='upper left')
    # plt.xlabel("time [s]")
    # plt.ylabel("$|\delta v|$")
    # plt.tight_layout()
    # if SAVE_FIGURES:
    #     plt.savefig(EXPERIMENT_NAME + "_verror.pdf")

    # plt.figure('Kd ddp', figsize=FEEDBACKFIGSIZE)
    # for i, ithSolverName in enumerate(solver_names):
    #     plt.plot(t_array, kd[:,i], colors[i], linewidth=4., label=ithSolverName)
    # # plt.legend(loc='upper left')
    # plt.xlabel("time [s]")
    # plt.ylabel("$| K_d |$")
    # plt.tight_layout()
    # if SAVE_FIGURES:
    #     plt.savefig(EXPERIMENT_NAME + "_Kd.pdf")

    # """ noise and feedback at impact """
    # CovarianceFIGSIZE = (15,15) 
    # for i, ithSolverName in enumerate(solver_names[1:]):
    #     plt.figure(figsize = CovarianceFIGSIZE)
    #     ax = sns.heatmap(solvers[i+1].measurement.runningDatas[43].covMatrix, center=0,cmap=color_map)
    #     ax.set_aspect("equal")
    #     ax.set_xticks(np.arange(2*solo12.nv))
    #     ax.set_yticks(np.arange(2*solo12.nv))
    #     ax.set_xticklabels([], rotation=90)
    #     ax.set_yticklabels([], rotation=0)
    #     # plt.title(ithSolverName+" Covariance")
    #     plt.tight_layout()
    #     if SAVE_FIGURES:
    #         plt.savefig(EXPERIMENT_NAME + "_"+ ithSolverName+"_Covariance.pdf")

    for i, ithSolverName in enumerate(solver_names):
        plt.figure(ithSolverName+"_KP_Contact_joints", figsize = (solo12_plots.nv, solo12_plots.m))
        ax = sns.heatmap(solvers[i].K[43][:,6:solo12.nv], center=0,cmap=color_map, robust=True)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(solo12.nv))
        ax.set_yticks(np.arange(12))
        ax.set_xticklabels([], rotation=90)
        ax.set_yticklabels([], rotation=0)
        # plt.title(ithSolverName+" KP")
        plt.tight_layout()
        if SAVE_FIGURES:
            plt.savefig(EXPERIMENT_NAME + "_"+ ithSolverName+"_KP_Contact_joints.pdf")
            
        plt.figure(ithSolverName+"_KD_Contact_joints", figsize = (solo12_plots.nv, solo12_plots.m))
        ax = sns.heatmap(solvers[i].K[43][:,solo12.nv+6:], center=0,cmap=color_map, robust=True)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(solo12.nv))
        ax.set_yticks(np.arange(12))
        ax.set_xticklabels([], rotation=90)
        ax.set_yticklabels([], rotation=0)
        # plt.title(ithSolverName+" KD")
        plt.tight_layout()
        if SAVE_FIGURES:
            plt.savefig(EXPERIMENT_NAME + "_"+ ithSolverName+"_KD_Contact_joints.pdf")

        plt.figure( ithSolverName+"_KP_Contact_base", figsize = (solo12_plots.nv, solo12_plots.m))
        ax = sns.heatmap(solvers[i].K[43][:,:6], center=0,cmap=color_map, robust=True)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(solo12.nv))
        ax.set_yticks(np.arange(12))
        ax.set_xticklabels([], rotation=90)
        ax.set_yticklabels([], rotation=0)
        # plt.title(ithSolverName+" KP")
        plt.tight_layout()
        if SAVE_FIGURES:
            plt.savefig(EXPERIMENT_NAME + "_"+ ithSolverName+"_KP_Contact_base.pdf")

        plt.figure(ithSolverName+"_KD_Contact_base", figsize = (solo12_plots.nv, solo12_plots.m))
        ax = sns.heatmap(solvers[i].K[43][:,solo12.nv:solo12.nv+6], center=0,cmap=color_map, robust=True)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(solo12.nv))
        ax.set_yticks(np.arange(12))
        ax.set_xticklabels([], rotation=90)
        ax.set_yticklabels([], rotation=0)
        # plt.title(ithSolverName+" KD")
        plt.tight_layout()
        if SAVE_FIGURES:
            plt.savefig(EXPERIMENT_NAME + "_"+ ithSolverName+"_KD_Contact_base.pdf")


    plt.show()
