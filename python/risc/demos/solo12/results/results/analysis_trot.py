import numpy as np 
import os, sys, time 
src_path = os.path.abspath('../../src/py_locomotion/')
sys.path.append(src_path)
import matplotlib.pyplot as plt 
import pinocchio as pin 
import robots, locomotion_tools, plotting_tools


if __name__=="__main__":
    # Load the Robot and Plotting Tools 
    contact_names = ['FL_ANKLE', 'FR_ANKLE', 'HL_ANKLE', 'HR_ANKLE']
    solo_path = os.path.abspath(
        '../../../robots/robot_properties/robot_properties_solo')
    solo12 = robots.load_solo12_pinocchio(solo_path)  # pin.RobotWrapper object
    solo12_plots = plotting_tools.RobotPlottingTools(solo12, contact_names) # plotting tools 
    solo12_gaits = locomotion_tools.QuadrupedGaits(solo12, *contact_names) # some other tools 


    optimized_results = np.load("trot_optimized.npz")
    print optimized_results.keys()
    simulated_results = np.load("trot_simulated.npz")
    print simulated_results.keys()

    riskXref = optimized_results['riskXs']
    riskUref = optimized_results['riskUs']
    riskKref = optimized_results['riskK']
    riskCov = optimized_results['riskCov']

    ddpXref = optimized_results['ddpXs']
    ddpUref = optimized_results['ddpUs']
    ddpKref = optimized_results['ddpK']

    """ 
        Below are some experiment results results
        states are stored as follows = [samples, solver, t, states]
        solvers = ['ddp', 'risk_sensitive']
        forces are stored as follows = [samples, solver, t, 3*contact_point] # 12 total for solo12 
    """

    states_sim = simulated_results['states']
    forces_sim = simulated_results['forces']
    disturbances_sim = simulated_results['disturbances']

    """
    collect some error data
    """
    """ some experiment parameters """
    dt = 1.e-2 
    horizon = riskXref.shape[0]
    t_array = dt*np.arange(horizon)
    print "Horizon = ", horizon
    nSamples = states_sim.shape[0]
    print "Total samples = ", nSamples

    """ Computing Tracking Errors """
    state_errors = np.zeros([nSamples, 2, horizon, 2*solo12.nv])
    # loop to compute 
    for ithSample in range(nSamples):
        for ithSolver in range(2):
            if ithSolver == 0:
                """ DDP reference errors """
                for t in range(horizon):
                    state_errors[ithSample, ithSolver, t, :] = \
                    np.resize(solo12_gaits.state.diff(states_sim[ithSample, ithSolver,t, :][:,None],ddpXref[t]), 2*solo12.nv)
            elif ithSolver == 1:
                """ Risk Sensitive errors """
                for t in range(horizon):
                    state_errors[ithSample, ithSolver, t, :] = \
                    np.resize(solo12_gaits.state.diff(states_sim[ithSample, ithSolver,t, :][:,None],riskXref[t]), 2*solo12.nv)
            else: 
                pass 
    """ q error norms & v errors norm """

    q_error_norms = np.zeros([nSamples, 2, horizon])
    v_error_norms = np.zeros([nSamples, 2, horizon])

    for ithSample in range(nSamples):
        for t in range(horizon):
            for ithSolver in range(2):
                q_error_norms[ithSample, ithSolver, t] = np.linalg.norm(state_errors[ithSample, ithSolver, t, :solo12.nv])
                v_error_norms[ithSample, ithSolver, t] = np.linalg.norm(state_errors[ithSample, ithSolver, t, solo12.nv:])

    """ plot error per state per norm  """
    joint_names = []
    for b in solo12_plots.branch_joints:
        for j in b:
            joint_names += [j]

    names_q_error = ['x', 'y', 'dz', 'qx', 'qy', 'qz'] + joint_names
    names_v_error = ['vx', 'vy', 'vz', 'wx', 'wy', 'wz'] + joint_names

    for i, qi in enumerate(names_q_error):
        plt.figure(qi+" state risk error")
        for ithSample in range(nSamples):
            if ithSample==0:
                plt.plot(t_array, state_errors[ithSample, 1, :, i],  label=qi)
            else:
                plt.plot(t_array, state_errors[ithSample, 1, :, i])
        plt.legend()
        plt.title(qi+" state risk error")

    plt.show()


            

