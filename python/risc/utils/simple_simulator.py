import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import fromListToVectorOfString, rotate
import os, sys, time 
from os.path import dirname, join
import yaml
from simple_simulator import build_euler_simulator



class SimpleSimulator(object):
    def __init__(self, robot, simulation_config=None):
        """" takes a pin.RobotWrapper() object with operational frames defining 
        the contact points """
        assert simulation_config is not None, ' configuration file is not specified '
        self.config_path = simulation_config
        # load configuration file
        self.conf = self.import_config_file(simulation_config)['sim']
        self.robot = robot 
        self.model = self.robot.model
        self.data = self.model.createData()
        self.contact_names = None 
        self.planar = None 
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.n = self.nq + self.nv
        self.m = self.model.njoints - 2 # don't count universe and base in pinocchio models
        self.contacts = []
        # simulation parameters 
        self.dt = None 
        self.n_steps = None 
        self.joint_friction = np.array([0.]*self.nv)  
        self.stiffness = None 
        self.damping = None
        self.friction_coeff = None 

        self.forward_dynamics_method = 2 # ABA is fastest, check consim source code 
        # fill the parameters from the yaml file 
        self.parse_simulation_config()
        # construct the simulator 
        
        try:
            self.simulator = build_euler_simulator(self.dt, self.n_steps, self.model, self.data,
                self.stiffness, self.damping, self.friction_coeff, self.forward_dynamics_method) 
        except:
            raise BaseException('could not construct simulator')
        #add contacts to simulator
        self.parse_contact_model()
        # add joint damping  
        self.simulator.set_joint_friction(
            np.resize(self.joint_friction, (self.nv, 1)))
        # control selection matrix 
        self.B = np.vstack([np.zeros([self.nv-self.m, self.m]),np.eye(self.m)])
        self.nc = len(self.contacts)

        self.obstacles = []

    def reset_state(self, x):
        self.simulator.reset_state(x[:self.nq], x[self.nq:self.nq + self.nv], True)

    def add_box(self, name, position, size, visual=None):
        """ adds a box object to both simulation and gepetto viewer 
        Args:
            name: string assigning name to object 
            position: 3d np.array describing cartesian position of object 
            size: 3d np.array describing width(x), depth(y), height(z) of the object 
            visual: robot wrapper object used to visualize the robot 
        """
        # add to the simulator 
        try:
            self.simulator.add_box(self.stiffness, self.damping,
                    self.friction_coeff, position[:], size[:])
        except:
            raise BaseException('Failed to add BoxObstacle to the simulator') 
        # load visual in gepetto viewer 
        color = [0.,0.,0.,1.]
        obj_pose = pin.SE3.Identity()
        offset = np.zeros(3)
        offset[2] = -.015
        obj_pose.translation = position[:] + offset
        if visual is not None:
            viewer = visual.viz.viewer
            viewer.gui.addBox("world/pinocchio/"+name, size[0], size[1], size[2], color)
            viewer.gui.setVisibility("world/pinocchio/"+name,"ON")
            viewer.gui.applyConfigurations(["world/pinocchio/"+name], [pin.SE3ToXYZQUATtuple(obj_pose)])
            viewer.gui.refresh()

    def clear_objects(self):
        pass 

    def integrate_step(self, u):
        self.simulator.step(self.torques(u))

    def read_state(self):
        return np.hstack([self.simulator.get_q(), self.simulator.get_v()])
    
    def read_contact_force(self):
        f = np.zeros([3, self.nc])
        # f = []
        for i,c in enumerate(self.contacts):
            f[:,i] = c.f.copy() # += [c.f.copy()]
        return f # np.resize(np.vstack(f), 3*self.nc)  

    def parse_contact_model(self):
        for name in self.contact_names:
            self.contacts += [self.simulator.add_contact_point(name, self.model.getFrameId(name), self.unilateral_contacts)]
            # self.contacts += [self.simulator.get_contact(name)]

    def import_config_file(self, file_name):
        self.path = os.path.abspath(file_name)
        with open(self.path, 'r') as file_descriptor:
            conf = yaml.load(file_descriptor, Loader=yaml.FullLoader)
        return conf

    def torques(self, u):
        """ returns control vector for underactuated system
        compatible with pinocchio"""
        return np.matrix(self.B.dot(u)).T

    def parse_simulation_config(self):
        self.dt = self.conf['dt']
        self.n_steps = self.conf['n_steps']
        self.stiffness = self.conf['stiffness']*np.ones(3)
        self.damping = self.conf['damping']*np.ones(3)
        self.friction_coeff = self.conf['static_friction_coefficient']
        self.contact_names = self.conf['contact_points']
        self.unilateral_contacts = self.conf["unilateral_contact"]
   
        