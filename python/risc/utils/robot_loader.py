""" load pinocchio RobotWrapper objects """

import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import fromListToVectorOfString






def load_solo12_pinocchio(model_path):
    """ model path is path to robot_properties_solo """
    urdf_path = '/urdf/solo12.urdf'
    mesh_path = fromListToVectorOfString([dirname(model_path)])
    robot = RobotWrapper.BuildFromURDF(
        model_path+urdf_path, mesh_path, pin.JointModelFreeFlyer())

    q = [0., 0., 0.32, 0., 0., 0., 1.] + 4 * [0., 0., 0.]
    q[8], q[9], q[11], q[12] = np.pi/4, -np.pi/2, np.pi/4,-np.pi/2
    q[14], q[15], q[17], q[18] = -np.pi/4, np.pi/2, -np.pi/4, np.pi/2
    q = np.array(q)
    
    pin.framesForwardKinematics(robot.model, robot.data, q)
    q[2] = .32 - robot.data.oMf[robot.model.getFrameId('FL_FOOT')].translation[2]
    pin.framesForwardKinematics(robot.model, robot.data, q)

    robot.q0.flat = q
    robot.model.referenceConfigurations["half_sitting"] = robot.q0
    robot.model.referenceConfigurations["reference"] = robot.q0
    robot.model.referenceConfigurations["standing"] = robot.q0

    robot.defaultState = np.concatenate([robot.q0, np.zeros(robot.model.nv)])
    return robot


