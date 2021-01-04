import numpy as np 
import os, sys, time 
src_path = os.path.abspath('../../../')
sys.path.append(src_path)
import matplotlib.pyplot as plt 
import pinocchio as pin 
import crocoddyl 

from utils import robot_loader, locomotion_tools, plotting_tools, measurement, visualize, simple_simulator 
from solvers import risk_sensitive_solver
import seaborn as sns 


EXPERIMENT_NAME="Solo12_jump_ddp_vs_risk"
SAVE_FIGURES =False   
SMALL_SIZE = 8
MEDIUM_SIZE = 30
BIGGER_SIZE = 35

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
color_map = "BrBG"


# some parameters for the gait, solver and simulator 
timeStep=1.e-2 
sensitivity = 10.
DISPLAY_DDP_SOLN = False     
DISPLAY_RISK_SOLN = False    
WHICH_SIMULATION = "ConSim"
NO_FEEDBACK = False 

DISPLAY_SIM = True 
WHICH_SOLVER = "RISK"             
WHICH_SOLVER = "FDDP" 
PLOT_PLANNER_REF = False    
DISPLAY_PLANNER_REF = True  
ADD_DISTURBANCE = False      
disturbance_position = [0., 0., .015+.015]
disturbance_position2 = [.5, -.1225, .0125+.015]
# [.19, .147, .0075]
# disturbance_orientation = p.getQuaternionFromEuler([0,0,0])

disturbance_size = [1.5, 1.5, .03]
disturbance_size2 = [.1, .1, .025]


contact_names = ['FL_ANKLE', 'FR_ANKLE', 'HL_ANKLE', 'HR_ANKLE']
solo_path = os.path.abspath('../../../../../../robot_properties_solo') 
plan_path = os.path.abspath("../planner/jump_ref/new")
noise_models = ["Uniform", "SwingJoints","Unconstrained", "Contact"]
noise_models = ["Contact"]
class FullStateMeasurement(object):
    def __init__(self, running_models, measurement_models):
        self.rmodels = running_models
        self.measurementModels = measurement_models
        self.runningDatas = []

        for t, mModel in enumerate(self.measurementModels):
            self.runningDatas += [mModel.createData()]

# modelPath = '../../../../../robot_properties_solo'
# urdf = modelPath+'/urdf/solo12.urdf'
# srdf = modelPath+'/srdf/solo.srdf'




DISPLAY = [DISPLAY_DDP_SOLN, DISPLAY_RISK_SOLN, DISPLAY_PLANNER_REF, DISPLAY_SIM]

if __name__ == "__main__":
    solo12 = robot_loader.load_solo12_pinocchio(solo_path)
    solo12_plots = plotting_tools.RobotPlottingTools(solo12, contact_names)

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
                ax[i].plot(t_array, vref[:-1,3*j+i], label=contact_names[j][:2]+label_direction[i]+'vel')
            ax[i].legend()
            ax[i].grid()
        fig.canvas.set_window_title('contact velocities')
        # com ref 
        plt.figure()
        for i in range(3):
            plt.plot(t_array, com_ref[:,i], label='com'+label_direction[i])
        plt.grid()
        plt.legend()
        plt.title('CoM Ref trajectory')

        orientation_names = ['qx', 'qy', 'qz', 'qw']
        plt.figure()
        for i in range(4):
            plt.plot(t_array, qref[:,i+3], label=orientation_names[i])
        plt.grid()
        plt.legend()
        plt.title('base Orientation ')


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
        except:
            raise BaseException('Gepetto viewer not initialized ! ')




    if DISPLAY_PLANNER_REF:
        print("Displaying KinoDyn Reference Trajectory")
        for qi in qref:
            solo12.display(qi)
            time.sleep(1.e-2)


    solo12_gaits = locomotion_tools.QuadrupedGaits(solo12, *contact_names)
    print("solo gaits work")

    x0 = np.hstack([qref[0], dqref[0]])
    # x0[2] -= solo12_gaits.ankle_offset

    solo12_gaits.WHICH_MEASUREMENT = "Contact"
    loco3dModels, runningMeasurements = solo12_gaits.createProblemKinoDynJump(x0, timeStep, 
                    contact_status_ref, qref, dqref, contact_positions_ref, vref, com_ref)
    kinoOptProblem = crocoddyl.ShootingProblem(x0, loco3dModels, loco3dModels[-1])

    """ FDDP Solver """
    print("Solving FDDP")
    fddp =  crocoddyl.SolverFDDP(kinoOptProblem)
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
        loco3dModels, runningMeasurements = solo12_gaits.createProblemKinoDynJump(x0, timeStep, 
                    contact_status_ref, qref, dqref, contact_positions_ref, vref, com_ref)
        kinoOptProblem = crocoddyl.ShootingProblem(x0, loco3dModels, loco3dModels[-1])
        measurementModels = measurement.MeasurementModels(kinoOptProblem.runningModels, runningMeasurements)
        risk_solvers_soln += [risk_sensitive_solver.RiskSensitiveSolver(kinoOptProblem, measurementModels, 
                                                                        sensitivity)]
        risk_solvers_soln[-1].callback = [crocoddyl.CallbackLogger(),
                                crocoddyl.CallbackVerbose()]
        # 
        xs_risk = [xi.copy() for xi in fddp.xs]
        us_risk = [ui.copy() for ui in fddp.us]
        # 
        _, _, converged = risk_solvers_soln[-1].solve(100, xs_risk, us_risk, True)
        print("Solving "+ risk_names_dff[i]+ " Completed")

    

    # # if DISPLAY_DDP_SOLN: 
    # #     for xi in fddp.xs:
    # #         solo12.display(xi[:solo12.nq])
    # #         time.sleep(1.e-2) 

    #     solo12_plots.solver_contact_forces(fddp)
    #     solo12_plots.leg_controls(fddp)
    #     solo12_plots.contact_positions(fddp.xs)


    # # np.save('jump_reference_states_ddp.npy', fddp.xs)
    # # np.save('jump_reference_controls_ddp.npy', fddp.us)
    # # np.save('jump_feedback_ddp.npy', fddp.K)
    # # np.save('jump_reference_states_risk.npy', risk_solver.xs)
    # # np.save('jump_reference_controls_risk.npy', risk_solver.us)
    # # np.save('jump_feedback_risk.npy', risk_solver.K)
    # # 
    """ ConSim """
    solvers = [fddp]+ risk_solvers_soln
    solver_names = ['DDP'] + ["Risk_"+ n for n in noise_models]
    if WHICH_SIMULATION == "ConSim":
        contact_config_path = 'linear_contact.yaml'
        sim = simple_simulator.SimpleSimulator(solo12, contact_config_path)
        # simulator runs at 1 ms while the trajectory is optimized at 10 ms 
        
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
                if ADD_DISTURBANCE and t == 125 :
                    sim.add_box("box01", np.array(disturbance_position), np.array(disturbance_size), None)    
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

    # base_tracking_fig = solo12_plots.compare_base_tracking(states, fddp.xs, names=solver_names)
    # contact_height_tracking_fig = solo12_plots.compare_frame_height(contact_names, states, fddp.xs, names=solver_names ,setlim=False)
    # contact_force_fig = solo12_plots.compare_simulation_froces(fddp, forces, dt=1.e-2, names=solver_names)

    # if SAVE_FIGURES:
    #     base_tracking_fig[0].savefig(EXPERIMENT_NAME+"_base_tracking.pdf")
    #     contact_height_tracking_fig[0].savefig(EXPERIMENT_NAME+"_contact_height_tracking.pdf")
    #     contact_directions = ["_fx.pdf", "_fy.pdf", "_fz.pdf"]
    #     for i,d in enumerate(contact_directions):
    #         contact_force_fig[i].savefig(EXPERIMENT_NAME+ d)



    # FEEDBACKFIGSIZE=(10,10)

    # N = len(errors[0])
    # t_array = 1.e-2 * np.arange(N)
    # colors = ['b', 'g-', 'r--', 'm-.', 'k:']
    # # colors = ['b', 'r', 'g', 'm-.', 'k:']

    # dq = np.zeros([N,len(solver_names)])
    # dv = np.zeros([N,len(solver_names)])
    # kp = np.zeros([N,len(solver_names)])
    # kd = np.zeros([N,len(solver_names)])
    # ##
    # basePosErr  = np.zeros([N,len(solver_names)])
    # baseVelErr  = np.zeros([N,len(solver_names)])
    # jointPosErr = np.zeros([N,len(solver_names)])
    # jointVelErr = np.zeros([N,len(solver_names)])
    # ## 
    # baseKp  = np.zeros([N,len(solver_names)])
    # baseKd  = np.zeros([N,len(solver_names)])
    # jointKp = np.zeros([N,len(solver_names)])
    # jointKd = np.zeros([N,len(solver_names)])
    # for i, ithSolverName in enumerate(solver_names):
    #     for t in range(N):
    #         dq[t,i] = np.linalg.norm(errors[i][t][:solo12.nv])
    #         dv[t,i] = np.linalg.norm(errors[i][t][solo12.nv:])
    #         kp[t,i] = np.linalg.norm(solvers[i].K[t][:,:solo12.nv])
    #         kd[t,i] = np.linalg.norm(solvers[i].K[t][:,solo12.nv:])
    #         #
    #         basePosErr[t,i]  = np.linalg.norm(errors[i][t][:6])
    #         baseVelErr[t,i]  = np.linalg.norm(errors[i][t][solo12.nv:solo12.nv+6])
    #         jointPosErr[t,i] = np.linalg.norm(errors[i][t][6:solo12.nv])
    #         jointVelErr[t,i] = np.linalg.norm(errors[i][t][solo12.nv+6:])
    #         # 
    #         baseKp[t,i]  = np.linalg.norm(solvers[i].K[t][:,:6])
    #         baseKd[t,i]  = np.linalg.norm(solvers[i].K[t][:,solo12.nv:solo12.nv+6])
    #         jointKp[t,i] = np.linalg.norm(solvers[i].K[t][:,6:solo12.nv])
    #         jointKd[t,i] = np.linalg.norm(solvers[i].K[t][:,solo12.nv+6:])

     
    # plt.figure('configuration error for ddp', figsize=FEEDBACKFIGSIZE)
    # for i, ithSolverName in enumerate(solver_names):
    #     plt.plot(t_array, dq[:,i], colors[i], linewidth=4., label=ithSolverName)
    # # plt.ylim(-.01, 0.5)
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

    # plt.figure('velocity error for ddp', figsize=FEEDBACKFIGSIZE)
    # for i, ithSolverName in enumerate(solver_names):
    #     plt.plot(t_array, dv[:,i], colors[i], linewidth=4., label=ithSolverName)
    # # plt.ylim(-.01, 0.5)
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

    # plt.figure('ImpedanceRatio', figsize=FEEDBACKFIGSIZE)
    # for i, ithSolverName in enumerate(solver_names):
    #     plt.plot(t_array, kp[:,i]/kd[:,i], colors[i], linewidth=4., label=ithSolverName)
    # plt.legend(loc='upper left')
    # plt.xlabel("time [s]")
    # plt.ylabel("$|K_p|/|K_d|$")
    # plt.tight_layout()
    # if SAVE_FIGURES:
    #     plt.savefig(EXPERIMENT_NAME + "ImpedanceRatio.pdf")

    # plt.figure('ImpedanceRatio', figsize=FEEDBACKFIGSIZE)
    # for i, ithSolverName in enumerate(solver_names):
    #     plt.plot(t_array, kp[:,i]/kd[:,i], colors[i], linewidth=4., label=ithSolverName)
    # plt.legend(loc='upper left')
    # plt.xlabel("time [s]")
    # plt.ylabel("$|K_p|/|K_d|$")
    # plt.tight_layout()
    # if SAVE_FIGURES:
    #     plt.savefig(EXPERIMENT_NAME + "ImpedanceRatio.pdf")

    # plt.figure('BasePosErr', figsize=FEEDBACKFIGSIZE)
    # for i, ithSolverName in enumerate(solver_names):
    #     plt.plot(t_array, basePosErr[:,i], colors[i], linewidth=4., label=ithSolverName)
    # plt.legend(loc='upper left')
    # plt.xlabel("time [s]", fontsize=35)
    # plt.ylabel("Base $|\delta q|$", fontsize=35)
    # plt.tight_layout()
    # if SAVE_FIGURES:
    #     plt.savefig(EXPERIMENT_NAME + "BasePosErr.pdf")

    # plt.figure('BaseVelErr', figsize=FEEDBACKFIGSIZE)
    # for i, ithSolverName in enumerate(solver_names):
    #     plt.plot(t_array, baseVelErr[:,i], colors[i], linewidth=4., label=ithSolverName)
    # plt.legend(loc='upper left')
    # plt.xlabel("time [s]", fontsize=35)
    # plt.ylabel("Base $|\delta v|$", fontsize=35)
    # plt.tight_layout()
    # if SAVE_FIGURES:
    #     plt.savefig(EXPERIMENT_NAME + "BaseVelErr.pdf")

    # plt.figure('JointPosErr', figsize=FEEDBACKFIGSIZE)
    # for i, ithSolverName in enumerate(solver_names):
    #     plt.plot(t_array, jointPosErr[:,i], colors[i], linewidth=4., label=ithSolverName)
    # # plt.legend(loc='upper left')
    # plt.xlabel("time [s]", fontsize=35)
    # plt.ylabel("Joints $|\delta q|$", fontsize=35)
    # plt.tight_layout()
    # if SAVE_FIGURES:
    #     plt.savefig(EXPERIMENT_NAME + "JointPosErr.pdf")

    # plt.figure('JointVelErr', figsize=FEEDBACKFIGSIZE)
    # for i, ithSolverName in enumerate(solver_names):
    #     plt.plot(t_array, jointVelErr[:,i], colors[i], linewidth=4., label=ithSolverName)
    # # plt.legend(loc='upper left')
    # plt.xlabel("time [s]", fontsize=35)
    # plt.ylabel("Joints $|\delta v|$", fontsize=35)
    # plt.tight_layout()
    # if SAVE_FIGURES:
    #     plt.savefig(EXPERIMENT_NAME + "JointVelErr.pdf")

    # plt.figure('BaseImpedanceRatio', figsize=FEEDBACKFIGSIZE)
    # for i, ithSolverName in enumerate(solver_names):
    #     plt.plot(t_array, baseKp[:,i]/baseKd[:,i], colors[i], linewidth=4., label=ithSolverName)
    # plt.legend(loc='upper left')
    # plt.xlabel("time [s]", fontsize=35)
    # plt.ylabel("$|K_p|/|K_d|$", fontsize=35)
    # plt.tight_layout()
    # if SAVE_FIGURES:
    #     plt.savefig(EXPERIMENT_NAME + "BaseImpedanceRatio.pdf")
    
    # plt.figure('JointImpedanceRatio', figsize=FEEDBACKFIGSIZE)
    # for i, ithSolverName in enumerate(solver_names):
    #     plt.plot(t_array, jointKp[:,i]/jointKd[:,i], colors[i], linewidth=4., label=ithSolverName)
    # # plt.legend(loc='upper left')
    # plt.xlabel("time [s]", fontsize=35)
    # plt.ylabel("$|K_p|/|K_d|$", fontsize=35)
    # plt.tight_layout()
    # if SAVE_FIGURES:
    #     plt.savefig(EXPERIMENT_NAME + "JointImpedanceRatio.pdf")

    # plt.figure('BaseStiffness', figsize=FEEDBACKFIGSIZE)
    # for i, ithSolverName in enumerate(solver_names):
    #     plt.plot(t_array, baseKp[:,i], colors[i], linewidth=4., label=ithSolverName)
    # # plt.legend(loc='upper left')
    # plt.xlabel("time [s]", fontsize=35)
    # plt.ylabel("$|K_p|$", fontsize=35)
    # plt.tight_layout()
    # if SAVE_FIGURES:
    #     plt.savefig(EXPERIMENT_NAME + "BaseStiffness.pdf")
    
    # plt.figure('BaseDamping', figsize=FEEDBACKFIGSIZE)
    # for i, ithSolverName in enumerate(solver_names):
    #     plt.plot(t_array, baseKd[:,i], colors[i], linewidth=4., label=ithSolverName)
    # # plt.legend(loc='upper left')
    # plt.xlabel("time [s]", fontsize=35)
    # plt.ylabel("$|K_d|$", fontsize=35)
    # plt.tight_layout()
    # if SAVE_FIGURES:
    #     plt.savefig(EXPERIMENT_NAME + "BaseDamping.pdf")


    # plt.figure('JointStiffness', figsize=FEEDBACKFIGSIZE)
    # for i, ithSolverName in enumerate(solver_names):
    #     plt.plot(t_array, jointKp[:,i], colors[i], linewidth=4., label=ithSolverName)
    # # plt.legend(loc='upper left')
    # plt.xlabel("time [s]", fontsize=35)
    # plt.ylabel("$|K_p|$", fontsize=35)
    # plt.tight_layout()
    # if SAVE_FIGURES:
    #     plt.savefig(EXPERIMENT_NAME + "JointStiffness.pdf")
    
    # plt.figure('JointDamping', figsize=FEEDBACKFIGSIZE)
    # for i, ithSolverName in enumerate(solver_names):
    #     plt.plot(t_array, jointKd[:,i], colors[i], linewidth=4., label=ithSolverName)
    # # plt.legend(loc='upper left')
    # plt.xlabel("time [s]", fontsize=35)
    # plt.ylabel("$|K_d|$", fontsize=35)
    # plt.tight_layout()
    # if SAVE_FIGURES:
    #     plt.savefig(EXPERIMENT_NAME + "JointDamping.pdf")


    # """ noise and feedback at impact """
    # impactTime = 149
    # CovarianceFIGSIZE = (15,15) 
    # for i, ithSolverName in enumerate(solver_names[1:]):
    #     plt.figure(figsize = CovarianceFIGSIZE)
    #     ax = sns.heatmap(solvers[i+1].measurement.runningDatas[impactTime].covMatrix, center=0,cmap=color_map)
    #     ax.set_aspect("equal")
    #     ax.set_xticks(np.arange(2*solo12.nv))
    #     ax.set_yticks(np.arange(2*solo12.nv))
    #     ax.set_xticklabels([], rotation=90)
    #     ax.set_yticklabels([], rotation=0)
    #     # plt.title(ithSolverName+" Covariance")
    #     plt.tight_layout()
    #     if SAVE_FIGURES:
    #         plt.savefig(EXPERIMENT_NAME + "_"+ ithSolverName+"_Covariance.pdf")

    # for i, ithSolverName in enumerate(solver_names):
    #     plt.figure(ithSolverName+"_KP_Contact_joints", figsize = (solo12_plots.nv, solo12_plots.m))
    #     ax = sns.heatmap(solvers[i].K[impactTime][:,6:solo12.nv], center=0,cmap=color_map)
    #     ax.set_aspect("equal")
    #     ax.set_xticks(np.arange(solo12.nv))
    #     ax.set_yticks(np.arange(12))
    #     ax.set_xticklabels([], rotation=90)
    #     ax.set_yticklabels([], rotation=0)
    #     # plt.title(ithSolverName+" KP")
    #     plt.tight_layout()
    #     if SAVE_FIGURES:
    #         plt.savefig(EXPERIMENT_NAME + "_"+ ithSolverName+"_KP_Contact_joints.pdf")
            
    #     plt.figure(ithSolverName+"_KD_Contact_joints", figsize = (solo12_plots.nv, solo12_plots.m))
    #     ax = sns.heatmap(solvers[i].K[impactTime][:,solo12.nv+6:], center=0,cmap=color_map)
    #     ax.set_aspect("equal")
    #     ax.set_xticks(np.arange(solo12.nv))
    #     ax.set_yticks(np.arange(12))
    #     ax.set_xticklabels([], rotation=90)
    #     ax.set_yticklabels([], rotation=0)
    #     # plt.title(ithSolverName+" KD")
    #     plt.tight_layout()
    #     if SAVE_FIGURES:
    #         plt.savefig(EXPERIMENT_NAME + "_"+ ithSolverName+"_KD_Contact_joints.pdf")

    #     plt.figure( ithSolverName+"_KP_Contact_base", figsize = (solo12_plots.nv, solo12_plots.m))
    #     ax = sns.heatmap(solvers[i].K[impactTime][:,:6], center=0,cmap=color_map)
    #     ax.set_aspect("equal")
    #     ax.set_xticks(np.arange(solo12.nv))
    #     ax.set_yticks(np.arange(12))
    #     ax.set_xticklabels([], rotation=90)
    #     ax.set_yticklabels([], rotation=0)
    #     # plt.title(ithSolverName+" KP")
    #     plt.tight_layout()
    #     if SAVE_FIGURES:
    #         plt.savefig(EXPERIMENT_NAME + "_"+ ithSolverName+"_KP_Contact_base.pdf")

    #     plt.figure(ithSolverName+"_KD_Contact_base", figsize = (solo12_plots.nv, solo12_plots.m))
    #     ax = sns.heatmap(solvers[i].K[impactTime][:,solo12.nv:solo12.nv+6], center=0,cmap=color_map)
    #     ax.set_aspect("equal")
    #     ax.set_xticks(np.arange(solo12.nv))
    #     ax.set_yticks(np.arange(12))
    #     ax.set_xticklabels([], rotation=90)
    #     ax.set_yticklabels([], rotation=0)
    #     # plt.title(ithSolverName+" KD")
    #     plt.tight_layout()
    #     if SAVE_FIGURES:
    #         plt.savefig(EXPERIMENT_NAME + "_"+ ithSolverName+"_KD_Contact_base.pdf")

    plt.show()

    # """ video recordings """
    # urdf_path = urdf 
    # model_path = modelPath +'/../../../..'

    # CameraTransform =[-1.5264389514923096,
    #                 -1.6716818809509277,
    #                 0.2570425570011139,
    #                 0.6916553378105164,
    #                 -0.23205697536468506,
    #                 -0.21380648016929626,
    #                 0.6496531367301941]

    # visualizer = visualize.Visualizer(showFloor=True, cameraTF=CameraTransform)

    # ddp_options = {'contact_names': contact_names, 
    #                 'robot_color': [0.,1.,0.,.5],
    #                 'force_color': [0., 1., 0., .5],
    #                 'force_radius': .002, 
    #                 'force_length': .028,
    #                 'cone_color': [0., 1., 0., .3],
    #                 'cone_length': .02,
    #                 'friction_coeff': .7, 
    #                 'wireframe_mode': "FILL"
    #                 }

    # risk_options = {'contact_names':contact_names, 
    #                 'robot_color': [.7,.7, .7,1.],
    #                 'force_color': [0., 0., 1., 1.],
    #                 'force_radius': .002, 
    #                 'force_length': .028,
    #                 'cone_color': [0., 0., 1., .4],
    #                 'cone_length': .02,
    #                 'friction_coeff': .7,
    #                 'wireframe_mode': "FILL"
    #                 }


    # ddp_viz = visualize.ConsimVisual("DDP", urdf_path, 
    #                     [model_path], pin.JointModelFreeFlyer(), 
    #                     visualizer, ddp_options)

    # ddp_viz.loadViewerModel()

    # risk_viz = visualize.ConsimVisual("Risk", urdf_path, 
    #                     [model_path],pin.JointModelFreeFlyer(), 
    #                     visualizer, risk_options)

    # risk_viz.loadViewerModel()

    # saving_path = os.getcwd() + "/jump-no-block/"+EXPERIMENT_NAME
    # for t in range(N):
    #     if ADD_DISTURBANCE and t == 125 :
    #         print("Adding Box")
    #         ddp_viz.add_box("box01", np.array(disturbance_position), np.array(disturbance_size))
    #     ddp_viz.display(states[0][t][:solo12.nq],forces[0][t])
    #     risk_viz.display(states[1][t][:solo12.nq],forces[1][t])
    #     time.sleep(1.e-2)
    #     visualizer.captureFrame(saving_path+"_{:03d}".format(t)+".png")

