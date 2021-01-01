""" a demo file to load solo12 in gepetto viewer, consim and crocoddyl """

import numpy as np 
import os, sys, time 
src_path = os.path.abspath('../../src/py_locomotion/')
sys.path.append(src_path)
import matplotlib.pyplot as plt 
import pinocchio as pin 
import crocoddyl 

import robots, plotting_tools, locomotion_tools, measurement 
import risk_sensitive_solver

import jump_conf as conf

class FullStateMeasurement(object):
    def __init__(self, running_models, measurement_models):
        self.rmodels = running_models
        self.measurementModels = measurement_models
        self.runningDatas = []

        for t, mModel in enumerate(self.measurementModels):
            self.runningDatas += [mModel.createData()]


if __name__=="__main__":

    contact_names = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']
    solo_path = os.path.abspath(
        '../../../robot_properties_solo')
    solo12 = robots.load_solo12_pinocchio(solo_path)
    solo12_plots = plotting_tools.RobotPlottingTools(solo12, contact_names)

    print("solo robot loaded with configuration space dimension = %s"%solo12.model.nq)

    # read the plan 
    

    qref, dqref, fref, vref, contact_positions_ref, \
    contact_status_ref, com_ref = locomotion_tools.parse_kindynamic_plan_slo12(conf.path, solo_path)

    print("kinodynamic plan parsed properly")


    t_array = conf.timeStep * np.arange(qref.shape[0])
    print("total kinodynopt horizon = %s"%qref.shape[0])
    label_direction = [' x ', ' y ', ' z ']
    """ Plot KinoDynamicPlanner Reference Trajectories """
    if conf.PLOT_PLANNER_REF:
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
        plt.title('CoM Ref trajectory')

    


    if conf.GEPETTO_VIEWER:
        try:
            solo12.initViewer(loadModel=True)
            print("solo loaded successfully")
            # cameraTF = [3., 3.68, 0.84, 0.2, 0.62, 0.72, 0.22]
            # solo12.viewer.gui.setCameraTransform(0, cameraTF)
            # backgroundColor = [1., 1., 1., 1.]
            # floorColor = [0.7, 0.7, 0.7, 1.]
            #   
            # window_id = solo12.viz.viewer.gui.getWindowID("python-pinocchio")
            # solo12.viz.viewer.gui.setBackgroundColor1(window_id, backgroundColor)
            # solo12.viz.viewer.gui.setBackgroundColor2(window_id, backgroundColor)
            solo12.display(x0[:solo12.nq])
        except:
            raise BaseException('Could not load model in Gepetto Viewer !')

    # print solo12.model.referenceConfigurations["standing"].shape 

    """ Setup the Locomotion Problem """
    solo12_gaits = locomotion_tools.QuadrupedGaits(solo12, *contact_names)
    # Four types = [None, "Uniform", "SwingJoints", "Contact","Unconstrained"]
    solo12_gaits.WHICH_MEASUREMENT = "Contact"
    x0 = np.hstack([qref[0], dqref[0]])
    x0[2] -= solo12_gaits.ankle_offset

    loco3dModels, measurementModels = solo12_gaits.createProblemKinoDynJump(x0, conf.timeStep, 
                    contact_status_ref, qref, dqref, contact_positions_ref, vref)
    

    kinoOptProblem = crocoddyl.ShootingProblem(x0, loco3dModels, loco3dModels[-1])

    """ FDDP Solver """
    print("Running DDP Solver")
    fddp =  crocoddyl.SolverFDDP(kinoOptProblem)
    fddp.setCallbacks(
            [crocoddyl.CallbackLogger(),
            crocoddyl.CallbackVerbose()])
    # set initial guess and solve using FDDP
    xs = [solo12.model.defaultState] * (fddp.problem.T + 1)
    us = [m.quasiStatic(d, solo12.model.defaultState) for m, d in list(zip(fddp.problem.runningModels, fddp.problem.runningDatas))]
    fddp.solve(xs, us, 1000, False, 0.1)


    print("Running Risk Sensitive Solver")
    """ risk sensitive  """
    measurementModel = FullStateMeasurement(kinoOptProblem.runningModels, measurementModels)
    risk_solver = risk_sensitive_solver.RiskSensitiveSolver(kinoOptProblem, measurementModel, conf.sensitivity)
    risk_solver.callback = [crocoddyl.CallbackLogger(),
                            crocoddyl.CallbackVerbose()]
    # 
    xs_risk = [xi.copy() for xi in fddp.xs]
    us_risk = [ui.copy() for ui in fddp.us]
    # 

    # print("set candidate")
    # risk_solver.setCandidate(xs_risk, us_risk, True)
    # print("initial calc")
    # risk_solver.calc()
    # print("backward pass")
    # risk_solver.backwardPass()
    

    xs_risk, us_risk, converged = risk_solver.solve(1000, xs_risk, us_risk, True)


    # plt.show()