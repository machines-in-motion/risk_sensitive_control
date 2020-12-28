import numpy as np 
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import fromListToVectorOfString, rotate
from os.path import dirname, join, abspath

import measurement 
import crocoddyl 
import matplotlib.pyplot as plt 
# add package path manually
import os
import sys
#
src_path = abspath('../../')
sys.path.append(src_path)


def load_sphere_model():
    urdf_path = abspath('sphere.urdf')
    mesh_path = abspath('../')
    robot = RobotWrapper.BuildFromURDF(
        urdf_path, mesh_path, pin.JointModelFreeFlyer())
    q = np.array([0., 0., 0., 0., 0., 0., 1.])
    robot.q0 = q # place robot at the origin 
    robot.model.referenceConfigurations["reference"] = robot.q0
    return robot


class ActuationModelSphere(crocoddyl.ActuationModelAbstract):
    def __init__(self, state):
        crocoddyl.ActuationModelAbstract.__init__(self, state, state.nv)
        self.S = np.eye(state.nv)

    def calc(self, data, x, u):
        data.tau = self.S * u
        

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        data.dtau_du = self.S


class FullStateMeasurement(object):
    def __init__(self, running_models):
        self.rmodels = running_models
        self.measurementModels = []
        self.runningDatas = []
        for t, rmodel in enumerate(running_models):
            ndx = rmodel.state.ndx
            # 1.e-5* np.random.rand(ndx,ndx),\
            sd, sn, md, mn = .01 * np.eye(ndx), 1.e-3 * np.eye(ndx),\
                .01 * np.eye(ndx), 1.e-5 * np.eye(ndx)
            self.measurementModels += [
                measurement.MeasurementModelFullState(rmodel, sd, sn, md, mn)]
            self.runningDatas += [self.measurementModels[-1].createData()]


def plot_tracking_solution(xs, us, ref):
    """ ref is 7dim vector of the se3 element with 
    quaternion representing orientation"""
    timeArray = 1.e-2 * np.arange(len(xs))
    labels = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
    control_names = ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']
    xs = np.array(xs)
    us = np.array(us)
    plt.figure('position tracking')
    for i in range(3):
        plt.plot(timeArray, xs[:, i], linewidth=1.2, label=labels[i]+'_act')
        plt.plot(timeArray, [ref[i]]*len(xs), '--',
                 linewidth=1.2, label=labels[i]+'_ref')
    plt.grid()
    plt.legend()
    plt.title('position tracking')

    plt.figure('orientation tracking')
    for i in range(3, 7):
        plt.plot(timeArray, xs[:, i], linewidth=1.2, label=labels[i]+'_act')
        plt.plot(timeArray, [ref[i]]*len(xs), '--',
                 linewidth=1.2, label=labels[i]+'_ref')
    plt.grid()
    plt.legend()
    plt.title('orientation tracking')

    plt.figure('linear control input')
    for i in range(3):
        plt.plot(timeArray[:-1], us[:, i],
                 linewidth=1.2, label=control_names[i])
    plt.grid()
    plt.legend()
    plt.title('linear control input')

    plt.figure('angular control input')
    for i in range(3, 6):
        plt.plot(timeArray[:-1], us[:, i],
                 linewidth=1.2, label=control_names[i])
    plt.grid()
    plt.legend()
    plt.title('angular control input')


class SE3ControlProblem(object):
    def __init__(self, model):
        self.rmodel = model
        self.rdata = model.createData()
        self.state = crocoddyl.StateMultibody(self.rmodel)
        self.nu = self.rmodel.nv
        self.actuation = crocoddyl.ActuationModelAbstract(self.state, 6)

        q0 = np.array(6*[0.]+[1.])

        self.rmodel.defaultState = np.concatenate(
            [q0, np.zeros(self.rmodel.nv)])
        self.fid = self.rmodel.getFrameId('root_joint')

    def createPositionControlProblem(self, targetPose, timeStep, horizon):
        q0 = self.rmodel.defaultState[:self.rmodel.nq]
        #
        se3Model = []
        framePlacement = crocoddyl.FramePlacement(self.fid, targetPose)
        for t in range(horizon):
            costModel = crocoddyl.CostModelSum(self.state, self.nu)
            # control regulation
            ctrlReg = crocoddyl.CostModelControl(self.state, self.nu)
            costModel.addCost("ctrlReg", ctrlReg, 1e-3)
            framePlacementCost = crocoddyl.CostModelFramePlacement(self.state,
                                                                   framePlacement, self.nu)
            costModel.addCost('tracking', framePlacementCost, 1.e4)
            dmodel = crocoddyl.DifferentialActionModelFreeFwdDynamics(
                self.state, self.actuation, costModel)
            se3Model += [
                crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)]

        terminalCost = crocoddyl.CostModelSum(self.state, self.nu)
        # control regulation
        ctrlReg = crocoddyl.CostModelControl(self.state, self.nu)
        terminalCost.addCost("ctrlReg", ctrlReg, 1e-3)
        framePlacementCost = crocoddyl.CostModelFramePlacement(self.state,
                                                               framePlacement, self.nu)
        terminalCost.addCost('tracking', framePlacementCost, 1.e4)
        dmodel = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self.state, self.actuation, terminalCost)
        terminalModel = crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)
        #
        problem = crocoddyl.ShootingProblem(self.rmodel.defaultState.copy(),
                                            se3Model, terminalModel)
        return problem
