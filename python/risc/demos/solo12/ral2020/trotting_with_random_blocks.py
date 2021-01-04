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


""" SOME PARAMETERS """

timeStep=1.e-2 
sensitivity = 10. 

N_SAMPLES = 1 # number of noise samples 

WHICH_SIMULATOR = "ConSim"      # options = ['ConSim', 'Bullet']
WHICH_EXPERIMENT = "trot"       # options = ['trot', 'jump']
WHICH_MEASUREMENT = "Contact"  # Four types = [None, "Uniform", "SwingJoints", "Contact","Unconstrained"]

VISUALIZE = False 

RESULTS_DIRECTORY = "../results/"           # Directory to save generated plots 
MIN_DIST_HEIGHT = .005
MAX_DIST_HEIGHT = .035
contact_config_path = '../../models/config/linear_contact.yaml'

SAVE_RESULTS = False 


contact_names = ['FL_ANKLE', 'FR_ANKLE', 'HL_ANKLE', 'HR_ANKLE']
solo_path = os.path.abspath('../../../../../../robot_properties_solo') 
plan_path = os.path.abspath("../planner/trot_ref")




if __name__ == "__main__":
    # Load the Robot and Plotting Tools 
    solo12 = robot_loader.load_solo12_pinocchio(solo_path)  # pin.RobotWrapper object
    solo12_plots = plotting_tools.RobotPlottingTools(solo12, contact_names) # plotting tools 
    solo12_gaits = locomotion_tools.QuadrupedGaits(solo12, *contact_names)  # ontop of crocoddyl to generate OCP objects 


    # load reference trajectories or experiment dependent parameters 

    if WHICH_EXPERIMENT == "trot":
        # array =  [time_max_height, x_landing, y_landing] per footstep in increasing time order 
        footstep_locations =   [[39, -.098, .151],
                                [40, .315, -.142],
                                [74, -.015, -.16],
                                [75, .4, .124],
                                [109, .102, .155],
                                [110, .512, -.14],
                                [144, .181, -.172],
                                [145, .59, .098],
                                [179, .299, .157],
                                [180, .71, -.14],
                                [214, .37, -.181],
                                [215, .772, .074],
                                [249, .399, .161],
                                [250, .805, -.145]] 
        

        qref, dqref, fref, vref, contact_positions_ref, \
        contact_status_ref, com_ref = locomotion_tools.parse_kindynamic_plan_slo12(plan_path, solo_path)

        x0 = np.hstack([qref[0], dqref[0]])

        loco3dModels, runningMeasurements = solo12_gaits.createProblemStateTracking(x0, timeStep, 
                        contact_status_ref, qref, dqref, contact_positions_ref, vref)
    
        shootingProblem = crocoddyl.ShootingProblem(x0, loco3dModels, loco3dModels[-1])
    elif WHICH_EXPERIMENT == "jump":
        footstep_locations = [[[],[],[],[]],
                             [[],[],[],[]]] 
        path = os.path.abspath("../planner/jump_ref")

        qref, dqref, fref, vref, contact_positions_ref, \
        contact_status_ref, com_ref = locomotion_tools.parse_kindynamic_plan_slo12(path, solo_path)


    elif WHICH_EXPERIMENT == "step":
        footstep_locations = [[[],[],[],[]],
                             [[],[],[],[]]] 
    else:
        raise Exception("Experiment Name Not Recognized !")

    
    max_n_disturbances = len(footstep_locations)


    """ FDDP Solver """
    fddp =  crocoddyl.SolverFDDP(shootingProblem)
    fddp.setCallbacks(
            [crocoddyl.CallbackLogger(),
            crocoddyl.CallbackVerbose()])
    # set initial guess and solve using FDDP
    xs = [solo12.defaultState] * (fddp.problem.T + 1)
    us = [m.quasiStatic(d, solo12.defaultState) for m, d in list(zip(fddp.problem.runningModels, fddp.problem.runningDatas))]
    fddp.solve(xs, us, 1000, False, 0.1)

    """ risk sensitive  """
    measurementModels = measurement.MeasurementModels(kinoOptProblem.runningModels, runningMeasurements)
    risk_solver = risk_sensitive_solver.RiskSensitiveSolver(shootingProblem, measurementModels, sensitivity)
    risk_solver.callback = [crocoddyl.CallbackLogger(),
                            crocoddyl.CallbackVerbose()]
    # 
    xs_risk = [xi.copy() for xi in fddp.xs]
    us_risk = [ui.copy() for ui in fddp.us]
    # 
    xs_risk, us_risk, converged = risk_solver.solve(1000, xs_risk, us_risk, True)

    solvers = []
    solvers += [fddp]
    solvers += [risk_solver]

    ref_forces = solo12_plots.get_reference_forces(risk_solver, dt=1.e-2)

    # arrays to save results 
    if SAVE_RESULTS:
        np.savez_compressed(RESULTS_DIRECTORY + WHICH_EXPERIMENT +"_" + "optimized", 
                            ddpXs = fddp.xs, riskXs = risk_solver.xs, 
                            ddpUs = fddp.us, riskUs = risk_solver.us,
                            ddpK=fddp.K, riskK = risk_solver.K, riskCov = risk_solver.cov, 
                            forcesRef=ref_forces)

    state_results = np.zeros([N_SAMPLES, len(solvers), len(xs), solo12.nq+solo12.nv])
    force_results = np.zeros([N_SAMPLES, len(solvers), len(xs), 3*len(contact_names)])
    disturbances = np.zeros([N_SAMPLES, 1 + max_n_disturbances, 7])  # [t, l, w, h, x, y, z] of each obstacles


    # RUN CONSIM SIMULATION 
    if WHICH_SIMULATOR == "ConSim":
        if VISUALIZE:
            try:
                solo12.initViewer(loadModel=True)
                cameraTF = [3., 3.68, 0.84, 0.2, 0.62, 0.72, 0.22]
                solo12.viewer.gui.setCameraTransform(0, cameraTF)
                backgroundColor = [1., 1., 1., 1.]
                floorColor = [0.7, 0.7, 0.7, 1.]
                #   
                window_id = solo12.viz.viewer.gui.getWindowID("python-pinocchio")
                solo12.viz.viewer.gui.setBackgroundColor1(window_id, backgroundColor)
                solo12.viz.viewer.gui.setBackgroundColor2(window_id, backgroundColor)
                solo12.display(x0[:solo12.nq])
            except:
                raise BaseException('Gepetto viewer not initialized ! ')
        
        
        

        for i_sample in range(N_SAMPLES):
            
            print("Running Sample %s"%(i_sample+1))

            ni_disturnances = np.random.randint(low=1,high=max_n_disturbances) 
            which_disturbances =  np.random.randint(low=1,high=max_n_disturbances, size=ni_disturnances) 

            print("WHICH DISTURBANCES \n", which_disturbances) 
            for kid in range(7):
                disturbances[i_sample, 0, kid] = ni_disturnances          # number of disturbances for the sample 
           
            for nid in range(1, 1+ni_disturnances):
                index_sample =  int(which_disturbances[nid-1])
                
                disturbances[i_sample, nid, 0] = footstep_locations[index_sample][0]                                # t_ith_dis 
                disturbances[i_sample, nid, 1] = .1                                                                 # l_ith_dis 
                disturbances[i_sample, nid, 2] = .1                                                                 # w_ith_dis 
                disturbances[i_sample, nid, 3] = np.random.uniform(low=MIN_DIST_HEIGHT, high=MAX_DIST_HEIGHT)       # h_ith_dis 
                disturbances[i_sample, nid, 4] = footstep_locations[index_sample][1]                                # x_ith_dis 
                disturbances[i_sample, nid, 5] = footstep_locations[index_sample][2]                                # y_ith_dis 
                disturbances[i_sample, nid, 6] = .5 * disturbances[i_sample, nid, 3]                                # z_ith_dis 




            

            for solver_index, solver in enumerate(solvers): 
                # reset simulator 
                sim = robots.PinocchioSimulator(solo12, contact_config_path)
                x0 = np.resize(solver.xs[0].copy(), solver.xs[0].shape[0])
                sim.reset_state(x0)
                if VISUALIZE:
                    solo12.display(x0[:solo12.nq, None])
                nodes = []
                for dind in range(ni_disturnances):
                    name = "obstacle_%s"%dind 
                    nodes += ["world/pinocchio/"+name]
                    posi =  disturbances[i_sample,dind+1, 4:7]
                    sizei = disturbances[i_sample,dind+1, 1:4]
                    if VISUALIZE:
                        sim.add_box(name, posi, sizei, solo12) 
                    else:
                        sim.add_box(name, posi, sizei)



                state_results[i_sample, solver_index, 0, :] = x0.copy()
                force_results[i_sample, solver_index, 0, :] = sim.read_contact_force()
                


                
                for t, ui in enumerate(solver.us):
                    for d in range(10):
                        # diff(xact, xref) = xref [-] xact --> on manifold 
                        # compute feedback
                        xref = solo12_gaits.interpolate_state(solver.xs[t], solver.xs[t+1], .1*d)
                        xact = sim.read_state()
                        diff = solo12_gaits.state.diff(xact, xref)
                        u = np.resize(ui + solver.K[t].dot(diff), sim.m)
                        # send control command 
                        sim.integrate_step(u)
                    # log state at optimization frequecy 
                    state_results[i_sample, solver_index, t, :] = sim.read_state()
                    force_results[i_sample, solver_index, t, :] = sim.read_contact_force()

                if VISUALIZE:
                    print('displaying simulation ')
                    for t, xi in enumerate(solver.xs):
                        solo12.display(state_results[i_sample, solver_index, t][:solo12.nq])
                        time.sleep(1.e-2) 
                    
                    # remove disturbance visuals 
                    time.sleep(5.)
                    for nodeName in nodes: 
                        solo12.viz.viewer.gui.deleteNode(nodeName, False)
                        solo12.viz.viewer.gui.refresh()

    if SAVE_RESULTS:
        np.savez_compressed(RESULTS_DIRECTORY + WHICH_EXPERIMENT +"_" + "simulated", 
                            states = state_results, forces = force_results, 
                            disturbances = disturbances)







    
