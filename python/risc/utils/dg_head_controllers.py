""" This code includes a Dynamic Graph Head Controller to Simulate/Run the precomputed plans """

import time 
import numpy as np
import pinocchio as pin 
from bullet_utils.env import BulletEnvWithGround
from robot_properties_solo.solo12wrapper import Solo12Robot, Solo12Config

from dynamic_graph_head import ThreadHead, SimHead, SimVicon, HoldPDController

class WholeBodyFeedbackController:
    def __init__(self, head, vicon_name, reference_path):
        self.robot = Solo12Config.buildRobotWrapper()
        self.rmodel = self.robot.model
        self.vicon_name = vicon_name
        # load precomputed trajectories 
        self.K = np.load(reference_path+'_K_ref.npy') 
        self.k = np.load(reference_path+'_u_ref.npy')  
        self.x_ref = np.load(reference_path+'_x_ref.npy')  
        # process trajectories 
        self.horizon = self.k.shape[0]
        self.x0 = self.x_ref[0]

        # read sensors 
        self.joint_positions = head.get_sensor('joint_positions')
        self.joint_velocities = head.get_sensor('joint_velocities')
        self.imu_gyroscope = head.get_sensor('imu_gyroscope')
        # some variables 
        self.x = np.zeros(self.robot.nq + self.robot.nv)
        self.u = np.zeros(self.robot.nv -6)
        self.d = 0. # interpolation step 
        self.t = 0
        self.runController = True 
        # saftey controller 
        self.endController = HoldPDController(head, 3., 0.05, False) 
        

    def interpolate(self, x1, x2, alpha):
        """ interpolate between states """
        x = np.zeros(self.rmodel.nq+self.rmodel.nv)
        x[:self.rmodel.nq] =  pin.interpolate(self.rmodel, x1[:self.rmodel.nq], x2[:self.rmodel.nq], alpha)
        x[self.rmodel.nq:] = x1[self.rmodel.nq:] + alpha*(x2[self.rmodel.nq:] - x1[self.rmodel.nq:])
        return x

    def difference(self, x1, x2):
        """ computes x2 (-) x1 on manifold """ 
        dx = np.zeros(2*self.rmodel.nv)
        dx[:self.rmodel.nv] = pin.difference(self.rmodel, x1[:self.rmodel.nq], x2[:self.rmodel.nq])
        dx[self.rmodel.nv:] =  x2[self.rmodel.nq:] -  x1[self.rmodel.nq:]
        return dx  

    def warmup(self, thread):
        thread.vicon.bias_position(self.vicon_name)

    def start_controller(self):
        self.runController = True 
        
    def get_base(self, thread):
        base_pos, base_vel = thread.vicon.get_state(self.vicon_name)
        base_vel[3:] = self.imu_gyroscope
        return base_pos, base_vel
    
    def run(self, thread):
        # get feedback signal 
        base_pos, base_vel = self.get_base(thread)
        self.x[:] = np.hstack([base_pos, self.joint_positions, base_vel, self.joint_velocities])
        # interpolate x desired 
        xdes = self.interpolate(self.x_ref[self.t], self.x_ref[self.t+1], self.d)
        # compute error signal and feedback control 
        dx = self.difference(self.x, xdes)
        self.u[:] = self.k[self.t] - self.K[self.t].dot(dx)
        # set control 
        head.set_control('ctrl_joint_torques', self.u)
        # increment time and interpolation steps  
        if self.runController:
            self.d += .1 
            if (self.d - 1.)**2 < 1.e-5: 
                self.t += 1
                self.d = 0. 
        

        # end controller once horizon is reached 
        if self.t == self.horizon:
            print("plan executed switching to safety controller !")
            head.switch_controllers(self.endController) 
        