import numpy as np 
import os, sys, time 
src_path = os.path.abspath('../../../src/py_locomotion/')
sys.path.append(src_path)
import matplotlib.pyplot as plt 
import pinocchio as pin 
import crocoddyl 
import matplotlib
import robots, locomotion_tools, plotting_tools, measurement 
import risk_sensitive_solver

import visualize 
from example_robot_data.robots_loader import getModelPath

EXPERIMENT_NAME="Solo12_trot_ddp_vs_risk"
SAVE_FIGURES =True  
SMALL_SIZE = 8
MEDIUM_SIZE = 30
BIGGER_SIZE = 22

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

matplotlib.rc('xtick', labelsize=35)     
matplotlib.rc('ytick', labelsize=35)

# some parameters for the gait, solver and simulator 
timeStep=1.e-2 
contact_names = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']
solo_path = os.path.abspath('../../../../robot_properties_solo')
results_path = os.path.abspath("../results/results/35mmDisturbances")



modelPath = '/opt/openrobots/share/example-robot-data/robots/solo_description/robots'
urdf = modelPath+'/solo12.urdf'
srdf = modelPath+'/srdf/solo.srdf'


if __name__=="__main__":
    solo12 = robots.load_solo12_pinocchio(solo_path)
    solo12_plots = plotting_tools.RobotPlottingTools(solo12, contact_names)
    solo12_gaits = locomotion_tools.QuadrupedGaits(solo12, *contact_names) # some other tools 

    optimized_results = np.load(results_path+"/trot_optimized.npz")
    print(optimized_results.keys())
    simulated_results = np.load(results_path+"/trot_simulated.npz")
    print(simulated_results.keys())

    riskXref = optimized_results['riskXs']
    riskUref = optimized_results['riskUs']
    riskKref = optimized_results['riskK']
    riskCov = optimized_results['riskCov']

    ddpXref = optimized_results['ddpXs']
    ddpUref = optimized_results['ddpUs']
    ddpKref = optimized_results['ddpK']
    forcesRef = optimized_results['forcesRef']

    """ 
        Below are some experiment results results
        states are stored as follows = [samples, solver, t, states]
        solvers = ['ddp', 'risk_sensitive']
        forces are stored as follows = [samples, solver, t, 3*contact_point] # 12 total for solo12 
    """

    states_sim = simulated_results['states']
    forces_sim = simulated_results['forces']
    disturbances_sim = simulated_results['disturbances']

    """ plot error per state per norm  """
    joint_names = []
    for b in solo12_plots.branch_joints:
        for j in b:
            joint_names += [j]

    names_q_error = ['x', 'y', 'dz', 'qx', 'qy', 'qz'] + joint_names
    names_v_error = ['vx', 'vy', 'vz', 'wx', 'wy', 'wz'] + joint_names

    horizon = riskXref.shape[0]
    print("Horizon = ", horizon)
    t_array = timeStep*np.arange(horizon)
    nSamples = states_sim.shape[0]
    print("Total samples = ", nSamples)

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

    np.savez_compressed("trot_simulation_errors", state_errors = state_errors, 
        q_error_norm = q_error_norms, 
        v_error_norm = v_error_norms)

    """ Compute terminal Base errors 
    errors = [nSamples, solvers, [base_position, base_orientation, base_linear_vel, base_angular_vel]]
    """

    accumelated_base_error = np.zeros([nSamples, 2, horizon, 4])

    for ithSample in range(nSamples):
        for ithSolver in range(2):
            for t in range(horizon):
                accumelated_base_error[ithSample, ithSolver,t,0] = np.linalg.norm(state_errors[ithSample, ithSolver, t, :3])
                accumelated_base_error[ithSample, ithSolver,t, 1] = np.linalg.norm(state_errors[ithSample, ithSolver, t, 3:7])
                accumelated_base_error[ithSample, ithSolver,t, 2] = np.linalg.norm(state_errors[ithSample, ithSolver, t, solo12.nv:solo12.nv + 3])
                accumelated_base_error[ithSample, ithSolver,t, 3] = np.linalg.norm(state_errors[ithSample, ithSolver, t, solo12.nv+3:solo12.nv+6])


    nfailed_ddp = 0
    nfailed_risk = 0
    success_index_risk = []
    success_index_ddp = []
    error_threshold = .3 # threshold on base velocity to consider as not a successful pass 
    for ithSample in range(nSamples):
        if np.any(np.isnan(accumelated_base_error[ithSample, 1,:,0])) or np.any(accumelated_base_error[ithSample, 1,:,2]>error_threshold):
            nfailed_risk +=1  
        else:
            success_index_risk += [ithSample]
        
        if np.any(np.isnan(accumelated_base_error[ithSample, 0,:,0])) or np.any(accumelated_base_error[ithSample, 0,:,2]>error_threshold):
            nfailed_ddp +=1  
        else:
            success_index_ddp += [ithSample]
        
    print("total ddp failed simulations = ", nfailed_ddp)
    print("total risk failed simulations = ", nfailed_risk)

    # here compute the mean and standard deviation 
    risk_stats = np.zeros([horizon, 2,4])
    ddp_stats = np.zeros([horizon, 2,4])

    for i in range(4):
        ddp_stats[:,0,i] = np.mean(accumelated_base_error[success_index_ddp, 0,:,i], axis = 0)  
        ddp_stats[:,1,i] = np.std(accumelated_base_error[success_index_ddp, 0,:,i], axis = 0) 
        risk_stats[:,0,i] = np.mean(accumelated_base_error[success_index_risk, 1,:,i], axis = 0) 
        risk_stats[:,1,i] = np.std(accumelated_base_error[success_index_risk, 1,:,i], axis = 0) 

        
    plot_horizon = horizon -2
    plot_opacity = 0.2


    # component_name = ['Position', 'Orientation', 'Linear Velocity', 'Angular Velocity']
    # for i in range(4):
    #     plotTitle = 'Base '+ component_name[i] + ' Error Norm'
    #     plt.figure(plotTitle, figsize=(12,8))
    #     plt.plot(t_array[:plot_horizon],risk_stats[:plot_horizon,0, i], 'g', alpha=1., 
    #             linewidth=4., label='Risk')
    #     plt.fill_between(t_array[:plot_horizon],risk_stats[:plot_horizon,0, i]-risk_stats[:plot_horizon,1, i], 
    #                     risk_stats[:plot_horizon,0, i]+risk_stats[:plot_horizon,1, i], alpha=.25, color='g')
    #     plt.plot(t_array[:plot_horizon],ddp_stats[:plot_horizon,0, i], 'b', alpha=1.,
    #             linewidth=4., label='DDP')
    #     plt.fill_between(t_array[:plot_horizon],ddp_stats[:plot_horizon,0, i]-ddp_stats[:plot_horizon,1, i], 
    #                     ddp_stats[:plot_horizon,0, i]+ddp_stats[:plot_horizon,1, i], alpha=.15, color='b')
    #     plt.legend(loc='upper left')
    #     plt.xlabel('time [s]')
    #     plt.ylabel(plotTitle)
    #     plt.tight_layout()
    #     if SAVE_FIGURES:
    #         plt.savefig(EXPERIMENT_NAME +'_' + plotTitle+".pdf")

    



    # """ Contact Force Statistics """
    # risk_normal_force_stats = np.zeros([horizon, len(contact_names),2])  
    # risk_tangential_force_stats = np.zeros([horizon, len(contact_names),2])  
    # ddp_normal_force_stats = np.zeros([horizon, len(contact_names),2])  
    # ddp_tangential_force_stats = np.zeros([horizon, len(contact_names),2])  

    
    # for i in range(len(contact_names)):
    #     ddp_normal_force_stats[:,i,0] =  np.mean(forces_sim[success_index_ddp, 0, :, 3*i+2], axis = 0)
    #     ddp_normal_force_stats[:,i,1] =  np.std(forces_sim[success_index_ddp, 0, :, 3*i+2], axis = 0)
    #     risk_normal_force_stats[:,i,0] =  np.mean(forces_sim[success_index_risk, 1, :, 3*i+2], axis = 0)
    #     risk_normal_force_stats[:,i,1] =  np.std(forces_sim[success_index_risk, 1, :, 3*i+2], axis = 0)
    #     #
    #     ddp_tan_force = np.sqrt(forces_sim[success_index_ddp, 0, :, 3*i]**2 + forces_sim[success_index_ddp, 0, :, 3*i+1]**2)
    #     risk_tan_force = np.sqrt(forces_sim[success_index_risk, 1, :, 3*i]**2 + forces_sim[success_index_risk, 1, :, 3*i+1]**2)
    #     #
    #     ddp_tangential_force_stats[:,i,0] =  np.mean(ddp_tan_force, axis = 0)
    #     ddp_tangential_force_stats[:,i,1] =  np.std(ddp_tan_force, axis = 0)
    #     risk_tangential_force_stats[:,i,0] =  np.mean(risk_tan_force, axis = 0)
    #     risk_tangential_force_stats[:,i,1] =  np.std(risk_tan_force, axis = 0)


    # fig, ax = plt.subplots(len(contact_names), 1, figsize=(15,15))
    # for i, cname in enumerate(contact_names):
    #     plotTitle =cname[:2] + ' $F_n$ [N]'
    #     ax[i].plot(t_array[:plot_horizon],risk_normal_force_stats[:plot_horizon,i,0], 'g', alpha=1., 
    #             linewidth=4., label='Risk')
    #     ax[i].fill_between(t_array[:plot_horizon],risk_normal_force_stats[:plot_horizon,i,0]-risk_normal_force_stats[:plot_horizon,i,1], 
    #                     risk_normal_force_stats[:plot_horizon,i,0]+risk_normal_force_stats[:plot_horizon,i,1], alpha=.25, color='g')
    #     ax[i].plot(t_array[:plot_horizon],risk_normal_force_stats[:plot_horizon,i,0], 'b', alpha=1., 
    #             linewidth=4., label='DDP')
    #     ax[i].fill_between(t_array[:plot_horizon],ddp_normal_force_stats[:plot_horizon,i,0]-ddp_normal_force_stats[:plot_horizon,i,1], 
    #                     ddp_normal_force_stats[:plot_horizon,i,0]+ddp_normal_force_stats[:plot_horizon,i,1], alpha=.15, color='b')
    #     ax[i].legend(loc='upper left')
    #     ax[i].set_xlabel('time [s]')
    #     ax[i].set_ylabel(plotTitle)
    #     ax[i].set_ylim([-1., 25.] )
    #     fig.tight_layout()
    #     if SAVE_FIGURES:
    #         plt.savefig(EXPERIMENT_NAME +'_normal_force_stat.pdf')


    # fig, ax = plt.subplots(len(contact_names), 1, figsize=(15,15))
    # for i, cname in enumerate(contact_names):
    #     plotTitle =cname[:2] + ' $F_t$ [N]'
    #     ax[i].plot(t_array[:plot_horizon],risk_tangential_force_stats[:plot_horizon,i,0], 'g', alpha=1., 
    #             linewidth=4., label='Risk')
    #     ax[i].fill_between(t_array[:plot_horizon],risk_tangential_force_stats[:plot_horizon,i,0]-risk_tangential_force_stats[:plot_horizon,i,1], 
    #                     risk_tangential_force_stats[:plot_horizon,i,0]+risk_tangential_force_stats[:plot_horizon,i,1], alpha=.25, color='g')
    #     ax[i].plot(t_array[:plot_horizon],ddp_tangential_force_stats[:plot_horizon,i,0], 'b', alpha=1., 
    #             linewidth=4., label='DDP')
    #     ax[i].fill_between(t_array[:plot_horizon],ddp_tangential_force_stats[:plot_horizon,i,0]-ddp_tangential_force_stats[:plot_horizon,i,1], 
    #                     ddp_tangential_force_stats[:plot_horizon,i,0]+ddp_tangential_force_stats[:plot_horizon,i,1], alpha=.15, color='b')
    #     ax[i].legend(loc='upper left')
    #     ax[i].set_xlabel('time [s]')
    #     ax[i].set_ylabel(plotTitle)
    #     ax[i].set_ylim([-1., 15.])
    #     fig.tight_layout()
    #     if SAVE_FIGURES:
    #         plt.savefig(EXPERIMENT_NAME +'_tangent_force_stat.pdf')

    # plt.show()


    """ video recordings """
    urdf_path = urdf 
    model_path = modelPath +'/../../../..'

    CameraTransform =[1.688956618309021,
                    -2.023496389389038,
                    0.4910387396812439,
                    0.6164828538894653,
                    0.17589177191257477,
                    0.191686749458313,
                    0.7431467771530151]

    visualizer = visualize.Visualizer(showFloor=True, cameraTF=CameraTransform)

    ddp_options = {'contact_names': contact_names, 
                    'robot_color': [0.,1.,0.,.5],
                    'force_color': [0., 1., 0., .5],
                    'force_radius': .002, 
                    'force_length': .028,
                    'cone_color': [0., 1., 0., .3],
                    'cone_length': .02,
                    'friction_coeff': .7, 
                    'wireframe_mode': "FILL"
                    }

    risk_options = {'contact_names':contact_names, 
                    'robot_color': [.7,.7, .7,1.],
                    'force_color': [0., 0., 1., 1.],
                    'force_radius': .002, 
                    'force_length': .028,
                    'cone_color': [0., 0., 1., .4],
                    'cone_length': .02,
                    'friction_coeff': .7,
                    'wireframe_mode': "FILL"
                    }


    ddp_viz = visualize.ConsimVisual("DDP", urdf_path, 
                        [model_path], pin.JointModelFreeFlyer(), 
                        visualizer, ddp_options)

    ddp_viz.loadViewerModel()

    risk_viz = visualize.ConsimVisual("Risk", urdf_path, 
                        [model_path],pin.JointModelFreeFlyer(), 
                        visualizer, risk_options)

    risk_viz.loadViewerModel()

    forces = np.zeros([3, 4])
    N = 300 
    samples = [success_index_risk[91], success_index_risk[92], success_index_risk[93], 
    success_index_risk[94], success_index_risk[95]] #[120,1, 3, 45, 69]
    saving_path = os.getcwd() + "/trot-vids/"+EXPERIMENT_NAME
    for ithSample in samples:
        ddp_viz.gui.refresh()
        nDist = int(disturbances_sim[ithSample, 0, 0])
        for i in range(nDist):
            name = "obstacle_%s"%i 
            posi =  disturbances_sim[ithSample,i+1, 4:7]
            sizei = disturbances_sim[ithSample,i+1, 1:4]
            ddp_viz.add_box(name, posi, sizei)
        for t in range(N):
            ddp_viz.display(states_sim[ithSample, 0,t, :solo12.nq],forces)
            risk_viz.display(states_sim[ithSample, 1,t, :solo12.nq],forces)
            time.sleep(1.e-2)
            visualizer.captureFrame(saving_path+"_%s"%ithSample+"_{:03d}".format(t)+".png")
        
        for i in range(nDist):
            name = "obstacle_%s"%i 
            ddp_viz.removeObject(name)
        # ddp_viz.gui.refresh()






    

