""" This code includes a Dynamic Graph Head Controller to Simulate/Run the precomputed plans """

import time 
import numpy as np
import pinocchio as pin 
from bullet_utils.env import BulletEnvWithGround
from robot_properties_solo.solo12wrapper import Solo12Robot, Solo12Config
import mim_control_cpp
from dynamic_graph_head import ThreadHead, SimHead, SimVicon, HoldPDController

class SliderPDController:
    def __init__(self, head, Kp, Kd):
        self.head = head
        self.joint_positions = head.get_sensor('joint_positions')
        self.joint_velocities = head.get_sensor('joint_velocities')

        self.slider_positions = head.get_sensor('slider_positions')

        self.scale = np.pi
        self.Kp = Kp
        self.Kd = Kd
        self.loop_counter = 0 
        self.max_log = 5000 # run pd for five seconds log and switch back off 

        self.sensor_q = np.zeros([self.max_log, 19])
        self.sensor_v = np.zeros([self.max_log, 18])
        self.sensor_u = np.zeros([self.max_log, 12])
        self.sensor_tau = np.zeros([self.max_log, 12])

        self.q = np.zeros(19)
        self.v = np.zeros(18)
        self.u2 = np.zeros(12) # sensed torque 
        self.u = np.zeros(12)
        self.tau = np.zeros(12)



    def map_sliders(self, sliders):
        sliders_out = np.zeros(12)
        slider_A = sliders[0]
        slider_B = sliders[1]
        for i in range(4):
            sliders_out[3 * i + 0] = slider_A
            sliders_out[3 * i + 1] = slider_B
            sliders_out[3 * i + 2] = 2. * (1. - slider_B)

            if i >= 2:
                sliders_out[3 * i + 1] *= -1
                sliders_out[3 * i + 2] *= -1

        # Swap the hip direction.
        sliders_out[3] *= -1
        sliders_out[9] *= -1

        return sliders_out

    def warmup(self, thread):
        self.zero_pos = self.map_sliders(self.slider_positions)

    def run(self, thread):
        # if self.loop_counter == self.max_log:
        #     self.dump_data()
        #     thread.switch_controller(ZeroTorquesController(thread.head))
        #     return 
        # else:
        #     self.q[7:] = thread.head.get_sensor('joint_positions')
        #     self.v[6:] = thread.head.get_sensor('joint_velocities')
        #     self.q[:7], self.v[:6] = thread.vicon.get_state()
        #     self.v[3:6] = thread.head.get_sensor('imu_gyroscope') 
        #     self.u2[:] = thread.head.get_sensor('joint_torques')
        #     self.sensor_q[self.loop_counter, :]   = self.q.copy() 
        #     self.sensor_v[self.loop_counter, :]   = self.v.copy()
        #     self.sensor_u[self.loop_counter, :]   = self.tau.copy()
        #     self.sensor_tau[self.loop_counter, :] = self.u2.copy()

        self.des_position = self.scale * (
            self.map_sliders(self.slider_positions) - self.zero_pos)

        self.tau[:] = self.Kp * (self.des_position - self.joint_positions) - self.Kd * self.joint_velocities
        head.set_control('ctrl_joint_torques', self.tau)


        self.loop_counter += 1 

    def dump_data(self):
        """ dumps all stored data from controller to numpy arrays """
        np.save('data/pd_q',   self.sensor_q)
        np.save('data/pd_v',   self.sensor_v)
        np.save('data/pd_tau_cmd', self.sensor_u)
        np.save('data/pd_tau_sen', self.sensor_tau)


class BalancePDController:
    def __init__(self, head, Kp, Kd, qdes):
        self.head = head
        self.joint_positions = head.get_sensor('joint_positions')
        self.joint_velocities = head.get_sensor('joint_velocities')

        self.scale = np.pi
        self.Kp = Kp
        self.Kd = Kd

        self.tau = np.zeros(12)
        self.q_des = qdes

    def warmup(self, thread):
        pass

    def run(self, thread):
        self.tau[:] = self.Kp * (self.q_des[6:] - self.joint_positions) - self.Kd * self.joint_velocities
        head.set_control('ctrl_joint_torques', self.tau)


class WholeBodyFeedbackController:
    def __init__(self,head, vicon_name, controller_name, path=''):
        """ 
        head: 
        vicon_name: 
        path: 
        """
        self.robot = Solo12Config.buildRobotWrapper()
        self.vicon_name = vicon_name
        # load precomputed trajectories 
        self.K = np.load(controller_name+'_K_ref') 
        self.k = np.load(controller_name+'_u_ref')  
        self.x_ref = np.load(controller_name+'_k_ref')  
        # process trajectories 
        self.horizon = self.k.shape[0]
        self.x0 = self.x_ref[0]

        # read sensors 
        self.joint_positions = head.get_sensor('joint_positions')
        self.joint_velocities = head.get_sensor('joint_velocities')
        self.imu_gyroscope = head.get_sensor('imu_gyroscope')
        # some variables 
        self.x = np.zeros(self.robot.nq + self.robot.nv)
        self.u = np.zeros(self.robotnv -6)

    def interpolate(self, x1, x2, alpha):
        """ interpolate between states """
        pass 

    def difference(self, x1, x2):
        """ computes x2 (-) x1 on manifold """ 
        pass 

    def warmup(self, thread):
        thread.vicon.bias_position(self.vicon_name)

    def get_base(self, thread):
        base_pos, base_vel = thread.vicon.get_state(self.vicon_name)
        base_vel[3:] = self.imu_gyroscope
        return base_pos, base_vel
    
    def run(self, thread):
        base_pos, base_vel = self.get_base(thread)

        self.x[:] = np.hstack([base_pos, self.joint_positions, base_vel, self.joint_velocities])
