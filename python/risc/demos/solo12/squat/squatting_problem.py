"""" we create a solo squatting problem here """

import numpy as np 
import crocoddyl 
import pinocchio as pin 
import os, sys, time 
src_path = os.path.abspath('../')
sys.path.append(src_path)
from utils import robot_loader, measurement 

def state_from_bullet(q, dq):
    xv = np.vstack([q,dq])
    dim = xv.shape[0]
    return np.resize(xv,dim)

def force_from_bullet(contact_ids, actve_ids, forces):
    f = np.zeros(12)
    for i, ind in enumerate(actve_ids):
        k = contact_ids.index(ind)
        f[3*k:3*k+3] = np.resize(forces[i][:3] ,3)
    return f
 
def parse_kindynamic_plan_slo12(path, solo_path):
    contact_names = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']
    solo12 = robot_loader.load_solo12_pinocchio(solo_path)
    q = np.loadtxt(path+"/quadruped_generalized_positions.dat", dtype=float)[:,1:] 
    dq = np.loadtxt(path+"/quadruped_generalized_velocities.dat", dtype=float)[:,1:] 
    f = np.loadtxt(path+"/quadruped_forces.dat", dtype=float)[:,1:] 
    x = np.loadtxt(path+"/quadruped_positions_abs.dat", dtype=float)[:,1:] 
    v = np.loadtxt(path+"/quadruped_velocities_abs.dat", dtype=float)[:,1:] 
    contact_activation = np.loadtxt(path+"/quadruped_contact_activation.dat", dtype=float)[:,1:] 
    # com = np.loadtxt(path+"/quadruped_com.dat", dtype=float)[:,1:] 
    contact_ids = [solo12.model.getFrameId(cnt) for cnt in contact_names]
    contact_positions = np.zeros([q.shape[0], 12])
    # contact_status = np.zeros([q.shape[0], 4])
    com = np.zeros([q.shape[0], 3])

    for t,qt in enumerate(q):
        pin.framesForwardKinematics(solo12.model,solo12.data, qt)
        com[t,:] = pin.centerOfMass(solo12.model,solo12.data, qt)
        for i, cid in enumerate(contact_ids):
            contact_positions[t,3*i:3*i+3] = solo12.data.oMf[cid].translation
    #         if f[t,3*i+2]>=1.e-5:
    #             if contact_status[t-50,i] < 1.e-5 and t>50:
    #                 contact_status[t,i] = 5.
    #             else:
    #                 contact_status[t,i] = 10.       
    #         else:
    #             if  v[t,3*i+2]<-1.1e-3 and contact_positions[t,3*i+2]<.05:
    #                 contact_status[t,i] = 5.
    #             else:
    #                 contact_status[t,i] = 0.

    return q[::10,:],dq[::10,:],f[::10,:], v[::10,:], contact_positions[::10,:], contact_activation[::10,:], com[::10,:] 



class QuadrupedSquatting(object):
    def __init__(self, robot, fl_name,fr_name, hl_name,hr_name):
        self.robot = robot 
        self.rmodel = self.robot.model 
        self.rdata = self.rmodel.createData()
        self.state = crocoddyl.StateMultibody(self.rmodel)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        # Getting the frame id for all the legs
        self.flFootId = self.rmodel.getFrameId(fl_name)
        self.frFootId = self.rmodel.getFrameId(fr_name)
        self.hlFootId = self.rmodel.getFrameId(hl_name)
        self.hrFootId = self.rmodel.getFrameId(hr_name)
        # this defines the walking pattern for now 
        # self.contact_ids = [self.hlFootId, self.flFootId, self.hrFootId, self.frFootId]
        self.contact_ids = [self.flFootId, self.frFootId, self.hlFootId, self.hrFootId]
        self.contact_names = [fl_name,fr_name, hl_name,hr_name]
        self.fpos0 = None
        # Defining default state
        q0 = self.rmodel.referenceConfigurations["standing"]
        self.rmodel.defaultState = np.concatenate([q0, np.zeros(self.rmodel.nv)])
        self.firstStep = True
        # Defining the friction coefficient and normal
        self.mu = 0.3
        self.nsurf =  np.eye(3)
        self.baumgarte = np.array([0., 50.]) # pd gains to stabalize the contact 
        self.walking_sequence = []
        self.log_plan = 'ON'
        self.fl_plan = []
        self.fl_plan = []
        self.fl_plan = []
        self.fl_plan = []
        self.com_plan = []

        self.fpos0 = None  
        self.ankle_offset = .015
        # Four types = [None, "Uniform", "SwingJoints", "Contact","Unconstrained"]
        self.WHICH_MEASUREMENT = "Contact"

    def createBalanceProblem(self, x0, timeStep, horizon): 
        loco3dModel = []
        measurementModels = []
        for t in range(horizon):
            costModel = crocoddyl.CostModelSum(self.state, self.actuation.nu)
            # CoM cost 
            # comTrack = crocoddyl.CostModelCoMPosition(self.state, CoMRef[t][:,None], self.actuation.nu)
            # costModel.addCost("comTrack", comTrack, 1.e+5)

            # contact model and cost 
            contactModel = crocoddyl.ContactModelMultiple(self.state, self.actuation.nu) 
            for i, frame_id in enumerate(self.contact_ids): 
                # footRef = FootPosRef[t,3*i:3*i+3].copy()
                # footRef[2] = -1.e-5
                pin.framesForwardKinematics(self.rmodel,self.rdata, x0[:self.rmodel.nq])
                cone_rotation = self.rdata.oMf[frame_id].rotation.T.dot(self.nsurf)
                xref = crocoddyl.FrameTranslation(frame_id, np.array([0., 0., 0.]))
                supportContactModel = crocoddyl.ContactModel3D(self.state, xref, self.actuation.nu, 
                                                                self.baumgarte)
                contactModel.addContact(self.rmodel.frames[frame_id].name + "_contact", supportContactModel)
                # friction cone  
                cone = crocoddyl.FrictionCone(cone_rotation, self.mu, 4, True)#, 0., 1000.)
                frameCone = crocoddyl.FrameFrictionCone(frame_id, cone)

                frictionCone = crocoddyl.CostModelContactFrictionCone(
                    self.state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub)),
                    frameCone , self.actuation.nu)
                costModel.addCost(self.rmodel.frames[frame_id].name + "_frictionCone", frictionCone, 1.e-2) 
               

            stateWeights =[1.e-1] * 3 + [1.e-1] * 3 + [1.e-2, 1.e-2, 1.e-2] * 4 # (self.rmodel.nv - 6)
            stateWeights += [1.e-1] * 6 + [1.e-2] * (self.rmodel.nv - 6)
            qdes = x0[:self.rmodel.nq].copy()
            dqdes = x0[self.rmodel.nq:].copy()
            # qdes[2] -= self.ankle_offset   
            state_ref = x0.copy()
            stateReg = crocoddyl.CostModelState(self.state,
                        crocoddyl.ActivationModelWeightedQuad(np.array(stateWeights)**2),
                        state_ref, self.actuation.nu)
            ctrlReg = crocoddyl.CostModelControl(self.state, self.actuation.nu)
            costModel.addCost("stateReg", stateReg, 1.5e+2)
            costModel.addCost("ctrlReg", ctrlReg, 1.e-3)
            # differential ocp model 
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, 
            self.actuation, contactModel, costModel, 0., True) 
            loco3dModel += [crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)]
            # 
            """ Creating the Measurement Models """
            if self.WHICH_MEASUREMENT is None:
                pass 
            elif self.WHICH_MEASUREMENT == "Uniform":
                state_diffusion = timeStep * np.eye(dmodel.state.ndx)
                state_noise = np.eye(dmodel.state.ndx)
                measurement_diffusion = 10* timeStep * np.eye(dmodel.state.ndx)
                measurement_noise = np.eye(dmodel.state.ndx) 
                measurementMod = measurement.MeasurementModelFullState(loco3dModel[-1],state_diffusion, 
                            state_noise, measurement_diffusion, measurement_noise)
                measurementModels += [measurementMod]

            else:
                raise BaseException("Measurement Model Not Recognized")
        return loco3dModel, measurementModels

    
    def interpolate_state(self, x1, x2, d):
        """ interpolate state for feedback at higher rate that plan """
        x = np.zeros(self.rmodel.nq+self.rmodel.nv)
        x[:self.rmodel.nq] =  pin.interpolate(self.rmodel, x1[:self.rmodel.nq], x2[:self.rmodel.nq], d)
        x[self.rmodel.nq:] = x1[self.rmodel.nq:] + d*(x2[self.rmodel.nq:] - x1[self.rmodel.nq:])
        return x


