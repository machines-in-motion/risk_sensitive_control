''' basic tools to generate locomotion gaits '''

import numpy as np 
import crocoddyl 
import pinocchio as pin 
import os, sys, time 
src_path = os.path.abspath('../../src/py_locomotion/')
sys.path.append(src_path)
import robots, measurement 

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
    solo12 = robots.load_solo12_pinocchio(solo_path)
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



class QuadrupedGaits(object):
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
        self.mu = 0.7
        self.nsurf = np.array([0., 0., 1.])
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


    def createProblemStateTracking(self, x0, timeStep, ContactPlan, qRef, dqRef, FootPosRef, FootVelRef):
        horizon = ContactPlan.shape[0]
        loco3dModel = []
        measurementModels = []
        for t in range(horizon):
            costModel = crocoddyl.CostModelSum(self.state, self.actuation.nu)
            # create CoM Task 
            # comTrack = crocoddyl.CostModelCoMPosition(self.state, CoMRef[t][:,None], self.actuation.nu)
            # costModel.addCost("comTrack", comTrack, 1.e+5)
            # sort contact plan 
            support = []
            swing = []
            pre_impact = []

            for i, st in enumerate(ContactPlan[t]):
                if st>.5: # and st < 10.5:
                    support += [i]
                else:
                    swing += [i]
                # elif st > 4. and st < 5.5:
                #     pre_impact += [i]
                # elif st < 1.e-3:
                #     swing += [i]
                # else:
                #     raise BaseException("Contact Status not recognized")
            # create Contact Models and Costs 
            contactModel = crocoddyl.ContactModelMultiple(self.state, self.actuation.nu)
            supportIds = []
            if len(support)>0:
                for i in support:
                    frame_id  = self.contact_ids[i]
                    supportIds += [frame_id]
                    footRef = FootPosRef[t,3*i:3*i+3].copy()
                    footRef[2] -= self.ankle_offset
                    # xref = crocoddyl.FrameTranslation(frame_id, FootPosRef[t,3*i:3*i+3][:,None])
                    xref = crocoddyl.FrameTranslation(frame_id, np.array([0., 0., 0.]))
                    supportContactModel = crocoddyl.ContactModel3D(self.state, xref, self.actuation.nu, 
                                                                    self.baumgarte)
                    contactModel.addContact(self.rmodel.frames[frame_id].name + "_contact", 
                                            supportContactModel)
                    # friction cone  
                    cone = crocoddyl.FrictionCone(self.nsurf, self.mu, 4, False)
                    frameCone = crocoddyl.FrameFrictionCone(frame_id, cone)
                    frictionCone = crocoddyl.CostModelContactFrictionCone(
                        self.state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub)),
                        frameCone , self.actuation.nu)
                    costModel.addCost(self.rmodel.frames[frame_id].name + "_frictionCone", frictionCone, 2.e-3) 
            # state and control cost 
            # create swing cost 
            # if len(swing+pre_impact)>0:
            #     for i in swing+pre_impact:
            #         frame_id  = self.contact_ids[i]
            #         xref = crocoddyl.FrameTranslation(frame_id, FootPosRef[t,3*i:3*i+3][:,None]) 
            #         footTrack = crocoddyl.CostModelFrameTranslation(self.state, xref, self.actuation.nu)
            #         costModel.addCost(self.rmodel.frames[frame_id].name + "_footTrack", footTrack, 1.e-5) 
            #         motion_ref = pin.Motion.Zero()
            #         motion_ref.linear = np.resize(FootVelRef[t, 3*i:3*i+3] ,(3,1))
            #         vref = crocoddyl.FrameMotion(frame_id, motion_ref)
            #         FootVelCost = crocoddyl.CostModelFrameVelocity(self.state, vref, self.actuation.nu)
            #         costModel.addCost(self.rmodel.frames[frame_id].name + "_Vel", FootVelCost, 1.e-5)

            # if len(pre_impact)>0:
            #     for i in pre_impact:
            #         # position cost 
            #         frame_id  = self.contact_ids[i]
            #         # xref = crocoddyl.FrameTranslation(frame_id, FootPosRef[t,3*i:3*i+3][:,None]) 
            #         # footTrack = crocoddyl.CostModelFrameTranslation(self.state, xref, self.actuation.nu)
            #         # costModel.addCost(self.rmodel.frames[frame_id].name + "_footTrack", footTrack, 1.e+2) 
            #         # velocity cost 
            #         # vref = crocoddyl.FrameMotion(frame_id, pin.Motion.Zero())
            #         # impulseFootVelCost = crocoddyl.CostModelFrameVelocity(self.state, vref, self.actuation.nu)
            #         # costModel.addCost(self.rmodel.frames[frame_id].name + "_impulseVel", impulseFootVelCost, 1.e-3)
            #         motion_ref = pin.Motion.Zero()
            #         motion_ref.linear = np.resize(FootVelRef[t, 3*i:3*i+3] ,(3,1))
            #         vref = crocoddyl.FrameMotion(frame_id, motion_ref)
            #         FootVelCost = crocoddyl.CostModelFrameVelocity(self.state, vref, self.actuation.nu)
            #         costModel.addCost(self.rmodel.frames[frame_id].name + "_Vel", FootVelCost, 1.e-5)


            stateWeights = np.array([1.e-1] * 3 + [1.e-1] * 3 + [1.e-1] * (self.rmodel.nv - 6) + [1.e-3] * 6 
                                + [1.e-3] * (self.rmodel.nv - 6))
            qdes = qRef[t].copy()
            qdes[2] -= self.ankle_offset   
            state_ref = np.hstack([qdes, dqRef[t]])
            stateReg = crocoddyl.CostModelState(self.state,
                        crocoddyl.ActivationModelWeightedQuad(np.array(stateWeights**2)),
                        state_ref, self.actuation.nu)
            ctrlReg = crocoddyl.CostModelControl(self.state, self.actuation.nu)
            costModel.addCost("stateReg", stateReg, 1.e+5)
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
                state_diffusion = .1 * np.eye(dmodel.state.ndx)
                state_noise = 1.e-7 * np.eye(dmodel.state.ndx)
                measurement_diffusion = .1 * np.eye(dmodel.state.ndx)
                measurement_noise = 1.e-2 * np.eye(dmodel.state.ndx) 
                swingIds = None 
                measurementMod = measurement.MeasurementModelSwingJoints(loco3dModel[-1],state_diffusion, 
                            state_noise, measurement_diffusion, measurement_noise, self.contact_names, swingIds)
                measurementModels += [measurementMod]

            elif self.WHICH_MEASUREMENT == "SwingJoints":
                state_diffusion = .1 * np.eye(dmodel.state.ndx)
                state_noise = 1.e-6 * np.eye(dmodel.state.ndx)
                measurement_diffusion = .1 * np.eye(dmodel.state.ndx)
                measurement_noise = 5.e-3 * np.eye(dmodel.state.ndx) 
                swingIds = [self.contact_ids[i] for i in swing+pre_impact]
                if len(swingIds) == 0:
                    swingIds = None 
                    measurementMod = measurement.MeasurementModelSwingJoints(loco3dModel[-1],state_diffusion, 
                            state_noise, measurement_diffusion, measurement_noise, self.contact_names, swingIds)
                else:
                    swingQ_noise = [[5.e-1, 5.e-1, 5.e-1] for _ in swingIds]
                    swingdQ_noise = [[5.e-1, 5.e-1, 5.e-1] for _ in swingIds]
                    measurementMod = measurement.MeasurementModelSwingJoints(loco3dModel[-1],state_diffusion, 
                            state_noise, measurement_diffusion, measurement_noise, self.contact_names, swingIds,swingQ_noise, swingdQ_noise)
                measurementModels += [measurementMod]
            elif self.WHICH_MEASUREMENT == "Unconstrained":
                state_diffusion = .1 * np.eye(loco3dModel[-1].state.ndx)
                state_noise = 1.e-6 * np.eye(loco3dModel[-1].state.ndx)
                measurement_diffusion = .1 * np.eye(loco3dModel[-1].state.ndx)
                measurement_noise = 1.e-3 * np.eye(loco3dModel[-1].state.ndx) 
                swingIds = [self.contact_ids[i] for i in swing+pre_impact]
                if len(swingIds) == 0:
                    swingIds = None 
                    measurementMod = measurement.MeasurementModelSwingJoints(loco3dModel[-1],state_diffusion, 
                            state_noise, measurement_diffusion, measurement_noise, self.contact_names, swingIds)
                else:
                    swingQ_noise = [[5.e-1, 5.e-1, 5.e-1] for _ in swingIds]
                    swingdQ_noise = [[1.e-1, 1.e-1, 1.e-1] for _ in swingIds]
                    measurementMod = measurement.MeasurementModelContactNoise(loco3dModel[-1],state_diffusion, 
                            state_noise, measurement_diffusion, measurement_noise, swingIds ,swingQ_noise, swingdQ_noise)
                measurementModels += [measurementMod]
            elif self.WHICH_MEASUREMENT == "Contact":
                state_diffusion = .1 * np.eye(dmodel.state.ndx)
                state_noise = 1.e-7 * np.eye(dmodel.state.ndx)
                measurement_diffusion = .1 * np.eye(dmodel.state.ndx)
                measurement_noise = 1.e-3 * np.eye(dmodel.state.ndx) 
                swingIds = [self.contact_ids[i] for i in swing+pre_impact]
                if len(swingIds) == 0:
                    swingIds = None 
                    measurementMod = measurement.MeasurementModelSwingJoints(loco3dModel[-1],state_diffusion, 
                            state_noise, measurement_diffusion, measurement_noise, self.contact_names, swingIds)
                else:
                    swingQ_noise = [[5.e-1, 5.e-1, 5.e-1] for _ in swingIds]
                    swingdQ_noise = [[1.e-3, 1.e-3, 1.e-3] for _ in swingIds]
                    measurementMod = measurement.MeasurementModelContactConsistent(loco3dModel[-1],state_diffusion, 
                            state_noise, measurement_diffusion, measurement_noise, swingIds, supportIds ,swingQ_noise, swingdQ_noise)
                measurementModels += [measurementMod]
            else:
                raise BaseException("Measurement Model Not Recognized")
        return loco3dModel, measurementModels
            

    def createProblemKinoDynJump(self, x0, timeStep, ContactPlan, qRef, dqRef, FootPosRef, FootVelRef, CoMRef):
        horizon = qRef.shape[0]
        loco3dModel = []
        measurementModels = []
        for t in range(horizon):
            costModel = crocoddyl.CostModelSum(self.state, self.actuation.nu)
            # create CoM Task 
            comTrack = crocoddyl.CostModelCoMPosition(self.state, crocoddyl.ActivationModelQuad(3) ,CoMRef[t], self.actuation.nu)
            costModel.addCost("comTrack", comTrack, 1.e-5)
            # sort contact plan 
            support = []
            swing = []
            pre_impact = []

            for i, st in enumerate(ContactPlan[t]):
                if st>.5: # and st < 10.5:
                    support += [i]
                else:
                    swing += [i]
                # elif st > 4. and st < 5.5:
                #     pre_impact += [i]
                # elif st < 1.e-3:
                #     swing += [i]
                # else:
                #     raise BaseException("Contact Status not recognized")
            # create Contact Models and Costs 
            contactModel = crocoddyl.ContactModelMultiple(self.state, self.actuation.nu)
            supportIds = []
            if len(support)>0:
                for i in support:
                    frame_id  = self.contact_ids[i]
                    supportIds += [frame_id]
                    # xref = crocoddyl.FrameTranslation(frame_id, FootPosRef[t,3*i:3*i+3])
                    xref = crocoddyl.FrameTranslation(frame_id, np.array([0., 0., 0.]))
                    supportContactModel = crocoddyl.ContactModel3D(self.state, xref, self.actuation.nu, 
                                                                    self.baumgarte)
                    contactModel.addContact(self.rmodel.frames[frame_id].name + "_contact", 
                                            supportContactModel)
                    # friction cone  
                    cone = crocoddyl.FrictionCone(self.nsurf, self.mu, 4, False)
                    frictionCone = crocoddyl.CostModelContactFrictionCone(
                    self.state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub)),
                    crocoddyl.FrameFrictionCone(frame_id, cone), self.actuation.nu)
                    costModel.addCost(self.rmodel.frames[frame_id].name + "_frictionCone", frictionCone, 1.e-2) 
                    motion_ref = pin.Motion.Zero()
                    vref = crocoddyl.FrameMotion(frame_id, motion_ref)
                    FootVelCost = crocoddyl.CostModelFrameVelocity(self.state, vref, self.actuation.nu)
                    costModel.addCost(self.rmodel.frames[frame_id].name + "_Vel", FootVelCost, 1.e-3)
                    xref = crocoddyl.FrameTranslation(frame_id, FootPosRef[t,3*i:3*i+3]) 
                    footTrack = crocoddyl.CostModelFrameTranslation(self.state, xref, self.actuation.nu)
                    costModel.addCost(self.rmodel.frames[frame_id].name + "_footTrack", footTrack, 1.e-3) 
            # state and control cost 
            # create swing cost 
            # if len(support+pre_impact)>0: # t>120:# and t<180: # and len(pre_impact)>0:
            #     for i in support+pre_impact:
            #         frame_id  = self.contact_ids[i]
            #         motion_ref = pin.Motion.Zero()
            #         # motion_ref.linear = np.resize(FootVelRef[t, 3*i:3*i+3] ,(3,1))
            #         vref = crocoddyl.FrameMotion(frame_id, motion_ref)
            #         FootVelCost = crocoddyl.CostModelFrameVelocity(self.state, vref, self.actuation.nu)
            #         costModel.addCost(self.rmodel.frames[frame_id].name + "_Vel", FootVelCost, 1.e-3)
            # if len(swing)>0:
            #     for i in swing:
            #         frame_id  = self.contact_ids[i]
            #         xref = crocoddyl.FrameTranslation(frame_id, FootPosRef[t,3*i:3*i+3]) 
            #         footTrack = crocoddyl.CostModelFrameTranslation(self.state, xref, self.actuation.nu)
            #         costModel.addCost(self.rmodel.frames[frame_id].name + "_footTrack", footTrack, 1.e-5) 
            #     print 'swing foot cost at t = %s '%t
            # if t>165:
            #     for i in range(4):
            #         frame_id  = self.contact_ids[i]
            #         xref = crocoddyl.FrameTranslation(frame_id, FootPosRef[t,3*i:3*i+3]) 
            #         footTrack = crocoddyl.CostModelFrameTranslation(self.state, xref, self.actuation.nu)
            #         costModel.addCost(self.rmodel.frames[frame_id].name + "_footTrack", footTrack, 5.e-5)
            #         motion_ref = pin.Motion.Zero()
            #         motion_ref.linear = FootVelRef[t, 3*i:3*i+3] 
            #         vref = crocoddyl.FrameMotion(frame_id, motion_ref)
            #         FootVelCost = crocoddyl.CostModelFrameVelocity(self.state, vref, self.actuation.nu)
            #         costModel.addCost(self.rmodel.frames[frame_id].name + "_Vel", FootVelCost, 1.e-3)
            if len(swing)>0:
                for i in swing:
                    frame_id  = self.contact_ids[i]
                    xref = crocoddyl.FrameTranslation(frame_id, FootPosRef[t,3*i:3*i+3]) 
                    footTrack = crocoddyl.CostModelFrameTranslation(self.state, xref, self.actuation.nu)
                    costModel.addCost(self.rmodel.frames[frame_id].name + "_footTrack", footTrack, 5.e-5)
                    # motion_ref = pin.Motion.Zero()
                    motion_ref.linear = FootVelRef[t, 3*i:3*i+3] 
                    vref = crocoddyl.FrameMotion(frame_id, motion_ref)
                    FootVelCost = crocoddyl.CostModelFrameVelocity(self.state, vref, self.actuation.nu)
                    costModel.addCost(self.rmodel.frames[frame_id].name + "_Vel", FootVelCost, 1.e-2)

            stateWeights = np.array([5.e-1] * 3 + [5.e-1] * 3 + [5.e-1] * (self.rmodel.nv - 6) + [1.e-2] * 6 
            + [1.e-3] * (self.rmodel.nv - 6))
            state_ref = np.hstack([qRef[t],dqRef[t]])
            stateReg = crocoddyl.CostModelState(self.state,
                        crocoddyl.ActivationModelWeightedQuad(np.array(stateWeights**2)),
                        state_ref, self.actuation.nu)
            ctrlReg = crocoddyl.CostModelControl(self.state, self.actuation.nu)
            costModel.addCost("stateReg", stateReg, 1.e+0)
            costModel.addCost("ctrlReg", ctrlReg, 1.e-4)
            # differential ocp model 
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, 
            self.actuation, contactModel, costModel, 0., True) 
            loco3dModel += [crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)]
            # 
            """ Creating the Measurement Models """
            if self.WHICH_MEASUREMENT is None:
                pass 
            elif self.WHICH_MEASUREMENT == "Uniform":
                state_diffusion = .1 * np.eye(dmodel.state.ndx)
                state_noise = 1.e-7 * np.eye(dmodel.state.ndx)
                measurement_diffusion = .1 * np.eye(dmodel.state.ndx)
                measurement_noise = 5.e-3* np.eye(dmodel.state.ndx) 
                swingIds = None 
                measurementMod = measurement.MeasurementModelSwingJoints(loco3dModel[-1],state_diffusion, 
                            state_noise, measurement_diffusion, measurement_noise, self.contact_names, swingIds)
                measurementModels += [measurementMod]

            elif self.WHICH_MEASUREMENT == "SwingJoints":
                state_diffusion = .1 * np.eye(dmodel.state.ndx)
                state_noise = 1.e-7 * np.eye(dmodel.state.ndx)
                measurement_diffusion = .1 * np.eye(dmodel.state.ndx)
                measurement_noise = 5.e-3 * np.eye(dmodel.state.ndx) 
                swingIds = [self.contact_ids[i] for i in swing+pre_impact]
                if len(swingIds) == 0:
                    swingIds = None 
                    measurementMod = measurement.MeasurementModelSwingJoints(loco3dModel[-1],state_diffusion, 
                            state_noise, measurement_diffusion, measurement_noise, self.contact_names, swingIds)
                else:
                    swingQ_noise = [[5.e-5, 5.e-5, 5.e-5] for _ in swingIds]
                    swingdQ_noise = [[1.e-5, 1.e-5, 1.e-5] for _ in swingIds]
                    measurementMod = measurement.MeasurementModelSwingJoints(loco3dModel[-1],state_diffusion, 
                            state_noise, measurement_diffusion, measurement_noise, self.contact_names, swingIds,swingQ_noise, swingdQ_noise)
                measurementModels += [measurementMod]
            elif self.WHICH_MEASUREMENT == "Unconstrained":
                state_diffusion = .1 * np.eye(loco3dModel[-1].state.ndx)
                state_noise = 1.e-7 * np.eye(loco3dModel[-1].state.ndx)
                measurement_diffusion = .1 * np.eye(loco3dModel[-1].state.ndx)
                measurement_noise = 1.e-3 * np.eye(loco3dModel[-1].state.ndx) 
                swingIds = [self.contact_ids[i] for i in swing+pre_impact]
                if len(swingIds) == 0:
                    swingIds = None 
                    measurementMod = measurement.MeasurementModelSwingJoints(loco3dModel[-1],state_diffusion, 
                            state_noise, measurement_diffusion, measurement_noise, self.contact_names, swingIds)
                else:
                    swingQ_noise = [[5.e-5, 5.e-5, 5.e-5] for _ in swingIds]
                    swingdQ_noise = [[1.e-5, 1.e-5, 1.e-5] for _ in swingIds]
                    measurementMod = measurement.MeasurementModelContactNoise(loco3dModel[-1],state_diffusion, 
                            state_noise, measurement_diffusion, measurement_noise, swingIds ,swingQ_noise, swingdQ_noise)
                measurementModels += [measurementMod]
            elif self.WHICH_MEASUREMENT == "Contact":
                state_diffusion = .1 * np.eye(loco3dModel[-1].state.ndx)
                state_noise = 1.e-7 * np.eye(loco3dModel[-1].state.ndx)
                measurement_diffusion = .1 * np.eye(loco3dModel[-1].state.ndx)
                measurement_noise = 1.e-3 * np.eye(loco3dModel[-1].state.ndx) 
                swingIds = [self.contact_ids[i] for i in swing+pre_impact]
                if len(swingIds) == 0:
                    swingIds = None 
                    measurementMod = measurement.MeasurementModelSwingJoints(loco3dModel[-1],state_diffusion, 
                            state_noise, measurement_diffusion, measurement_noise, self.contact_names, swingIds)
                else:
                    swingQ_noise = [[5.e-5, 5.e-5, 5.e-5] for _ in swingIds]
                    swingdQ_noise = [[1.e-5, 1.e-5, 1.e-5] for _ in swingIds]
                    measurementMod = measurement.MeasurementModelContactConsistent(loco3dModel[-1],state_diffusion, 
                            state_noise, measurement_diffusion, measurement_noise, swingIds, supportIds ,swingQ_noise, swingdQ_noise)
                measurementModels += [measurementMod]
            else:
                raise BaseException("Measurement Model Not Recognized")
        return loco3dModel, measurementModels



    def createProblemFromKinoDynPlanner(self, x0, timeStep, ContactPlan, CoMRef, FootPosRef, FootVelRef):
        horizon = ContactPlan.shape[0]
        loco3dModel = []
        measurementModels = []

        for t in range(horizon):
            costModel = crocoddyl.CostModelSum(self.state, self.actuation.nu)
            # create CoM Task 
            comTrack = crocoddyl.CostModelCoMPosition(self.state, crocoddyl.ActivationModelQuad(3) ,CoMRef[t], self.actuation.nu)
            costModel.addCost("comTrack", comTrack, 1.e+5)
            # sort contact plan 
            support = []
            swing = []
            pre_impact = []
            for i, st in enumerate(ContactPlan[t]):
                if st>9.5 and st < 10.5:
                    support += [i]
                elif st > 4. and st < 5.5:
                    pre_impact += [i]
                elif st < 1.e-3:
                    swing += [i]
                else:
                    raise BaseException("Contact Status not recognized")
            # create Contact Models and Costs 
            contactModel = crocoddyl.ContactModelMultiple(self.state, self.actuation.nu)
            supportIds = []
            if len(support)>0:
                for i in support:
                    frame_id  = self.contact_ids[i]
                    supportIds += [frame_id]
                    # xref = crocoddyl.FrameTranslation(frame_id, FootPosRef[t,3*i:3*i+3][:,None])
                    xref = crocoddyl.FrameTranslation(frame_id, np.array([0., 0., 0.]))
                    supportContactModel = crocoddyl.ContactModel3D(self.state, xref, self.actuation.nu, 
                                                                    self.baumgarte)
                    contactModel.addContact(self.rmodel.frames[frame_id].name + "_contact", 
                                            supportContactModel)
                    # friction cone  
                    cone = crocoddyl.FrictionCone(self.nsurf, self.mu, 4, False)
                    frictionCone = crocoddyl.CostModelContactFrictionCone(
                    self.state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub)),
                    crocoddyl.FrameFrictionCone(frame_id, cone), self.actuation.nu)
                    costModel.addCost(self.rmodel.frames[frame_id].name + "_frictionCone", frictionCone,  1.e-3) 

            # create swing cost 
            if len(swing)>0:
                for i in swing:
                    frame_id  = self.contact_ids[i]
                    xref = crocoddyl.FrameTranslation(frame_id, FootPosRef[t,3*i:3*i+3]) 
                    footTrack = crocoddyl.CostModelFrameTranslation(self.state, xref, self.actuation.nu)
                    costModel.addCost(self.rmodel.frames[frame_id].name + "_footTrack", footTrack, 1.e+2) 
                    # motion_ref = pin.Motion.Zero()
                    # motion_ref.linear = FootVelRef[t, 3*i:3*i+3] 
                    # vref = crocoddyl.FrameMotion(frame_id, motion_ref)
                    # FootVelCost = crocoddyl.CostModelFrameVelocity(self.state, vref, self.actuation.nu)
                    # costModel.addCost(self.rmodel.frames[frame_id].name + "_Vel", FootVelCost, 1.e+1)

        
            # create pre-contact cost 
            if len(pre_impact)>0:
                for i in pre_impact:
                    # position cost 
                    frame_id  = self.contact_ids[i]
                    xref = crocoddyl.FrameTranslation(frame_id, FootPosRef[t,3*i:3*i+3]) 
                    footTrack = crocoddyl.CostModelFrameTranslation(self.state, xref, self.actuation.nu)
                    costModel.addCost(self.rmodel.frames[frame_id].name + "_footTrack", footTrack, 1.e+2) 
                    # velocity cost 
                    # vref = crocoddyl.FrameMotion(frame_id, pin.Motion.Zero())
                    # impulseFootVelCost = crocoddyl.CostModelFrameVelocity(self.state, vref, self.actuation.nu)
                    # costModel.addCost(self.rmodel.frames[frame_id].name + "_impulseVel", impulseFootVelCost, 1.e-3)
                    # motion_ref = pin.Motion.Zero()
                    # # motion_ref.linear = np.resize(np.zeros(3) ,(3,1))
                    # vref = crocoddyl.FrameMotion(frame_id, motion_ref)
                    # FootVelCost = crocoddyl.CostModelFrameVelocity(self.state, vref, self.actuation.nu)
                    # costModel.addCost(self.rmodel.frames[frame_id].name + "_Vel", FootVelCost, 1.e+2)


            # state and control cost 
            stateWeights = np.array([1.e-1] * 3 + [1.e-1] * 3 + [1.e-1] * (self.rmodel.nv - 6) + [1.e-1] * 6 
                                + [0.] * (self.rmodel.nv - 6))
            stateReg = crocoddyl.CostModelState(self.state,
                        crocoddyl.ActivationModelWeightedQuad(np.array(stateWeights**2)),
                        self.rmodel.defaultState, self.actuation.nu)
            ctrlReg = crocoddyl.CostModelControl(self.state, self.actuation.nu)
            costModel.addCost("stateReg", stateReg, 1.e-1)
            costModel.addCost("ctrlReg", ctrlReg, 5.e-3)
            # differential ocp model 
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, 
            self.actuation, contactModel, costModel, 0., True) 
            loco3dModel += [crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)]
            # 
            """ Creating the Measurement Models """
            if self.WHICH_MEASUREMENT is None:
                pass 
            elif self.WHICH_MEASUREMENT == "Uniform":
                state_diffusion = .01 * np.eye(dmodel.state.ndx)
                state_noise = 1.e-7 * np.eye(dmodel.state.ndx)
                measurement_diffusion = .01 * np.eye(dmodel.state.ndx)
                measurement_noise = 1.e-5 * np.eye(dmodel.state.ndx) 
                swingIds = None 
                measurementMod = measurement.MeasurementModelSwingJoints(loco3dModel[-1],state_diffusion, 
                            state_noise, measurement_diffusion, measurement_noise, self.contact_names, swingIds)
                measurementModels += [measurementMod]

            elif self.WHICH_MEASUREMENT == "SwingJoints":
                state_diffusion = .1 * np.eye(dmodel.state.ndx)
                state_noise = 1.e-7 * np.eye(dmodel.state.ndx)
                measurement_diffusion = .1 * np.eye(dmodel.state.ndx)
                measurement_noise = 5.e-3 * np.eye(dmodel.state.ndx) 
                swingIds = [self.contact_ids[i] for i in swing+pre_impact]
                if len(swingIds) == 0:
                    swingIds = None 
                    measurementMod = measurement.MeasurementModelSwingJoints(loco3dModel[-1],state_diffusion, 
                            state_noise, measurement_diffusion, measurement_noise, self.contact_names, swingIds)
                else:
                    swingQ_noise = [[1.e-1, 1.e-1, 1.e-1] for _ in swingIds]
                    swingdQ_noise = [[5.e-1, 5.e-1, 5.e-1] for _ in swingIds]
                    measurementMod = measurement.MeasurementModelSwingJoints(loco3dModel[-1],state_diffusion, 
                            state_noise, measurement_diffusion, measurement_noise, self.contact_names, swingIds,swingQ_noise, swingdQ_noise)
                measurementModels += [measurementMod]
            elif self.WHICH_MEASUREMENT == "Unconstrained":
                state_diffusion = .1 * np.eye(loco3dModel[-1].state.ndx)
                state_noise = 1.e-6 * np.eye(loco3dModel[-1].state.ndx)
                measurement_diffusion = .1 * np.eye(loco3dModel[-1].state.ndx)
                measurement_noise = 5.e-3 * np.eye(loco3dModel[-1].state.ndx) 
                swingIds = [self.contact_ids[i] for i in swing+pre_impact]
                if len(swingIds) == 0:
                    swingIds = None 
                    measurementMod = measurement.MeasurementModelSwingJoints(loco3dModel[-1],state_diffusion, 
                            state_noise, measurement_diffusion, measurement_noise, self.contact_names, swingIds)
                else:
                    swingQ_noise = [[5.e-1, 5.e-1, 5.e-1] for _ in swingIds]
                    swingdQ_noise = [[1.e-1, 1.e-1, 1.e-1] for _ in swingIds]
                    measurementMod = measurement.MeasurementModelContactNoise(loco3dModel[-1],state_diffusion, 
                            state_noise, measurement_diffusion, measurement_noise, swingIds ,swingQ_noise, swingdQ_noise)
                measurementModels += [measurementMod]
            elif self.WHICH_MEASUREMENT == "Contact":
                state_diffusion = .01 * np.eye(dmodel.state.ndx)
                state_noise = 1.e-7 * np.eye(dmodel.state.ndx)
                measurement_diffusion = .01 * np.eye(dmodel.state.ndx)
                measurement_noise = 1.e-4 * np.eye(dmodel.state.ndx) 
                swingIds = [self.contact_ids[i] for i in swing+pre_impact]
                if len(swingIds) == 0:
                    swingIds = None 
                    measurementMod = measurement.MeasurementModelSwingJoints(loco3dModel[-1],state_diffusion, 
                            state_noise, measurement_diffusion, measurement_noise, self.contact_names, swingIds)
                else:
                    if t>165:# and t<180: # and len(pre_impact)>0:
                        measurement_noise = 1.e-4 * np.eye(dmodel.state.ndx) 
                        swingQ_noise = [[5.e-1, 5.e-1, 5.e-1] for _ in swingIds]
                        swingdQ_noise = [[1.e-1, 1.e-1, 1.e-1] for _ in swingIds]
                    else:
                        swingQ_noise = [[5.e-1, 5.e-1, 5.e-1] for _ in swingIds]
                        swingdQ_noise = [[1.e-1, 1.e-1, 1.e-1] for _ in swingIds]
                    measurementMod = measurement.MeasurementModelContactConsistent(loco3dModel[-1],state_diffusion, 
                            state_noise, measurement_diffusion, measurement_noise, swingIds, supportIds ,swingQ_noise, swingdQ_noise)
                measurementModels += [measurementMod]
            else:
                raise BaseException("Measurement Model Not Recognized")
        return loco3dModel, measurementModels


    def createBalanceProblem(self, x0, timeStep, supportKnots, stepKnots, comOffset):
        # Compute the current foot positions
        q0 = x0[:self.rmodel.nq]
        q0[2] += self.ankle_offset 
        pin.forwardKinematics(self.rmodel, self.rdata, q0)
        pin.updateFramePlacements(self.rmodel, self.rdata)

        self.fpos0 = [self.rdata.oMf[fid].translation for fid in self.contact_ids]
 
        comRef = pin.centerOfMass(self.rmodel, self.rdata, q0) 

        comInit = pin.centerOfMass(self.rmodel, self.rdata, q0) 
        # comRef[2] = np.asscalar(pin.centerOfMass(self.rmodel, self.rdata, q0)[2])
        loco3dModel = []       
        """ Full Support & Balance  """
        for t in range(supportKnots):
            dmodel = self.createFullSupportModel(comRef)
            loco3dModel += [crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)]

        """ Move CoM Down """
        for t in range(stepKnots):
            comRef[2] = comInit[2] -  t*comOffset/stepKnots
            dmodel = self.createFullSupportModel(comRef)
            loco3dModel += [crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)]

        """ stay down for a bit """
        for t in range(supportKnots):
            dmodel = self.createFullSupportModel(comRef)
            loco3dModel += [crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)]

        comInt = comRef.copy()
        """ go up double  as fast """
        for t in range(stepKnots):
            comRef[2] = comInit[2] +  2.*t*comOffset/stepKnots
            dmodel = self.createFullSupportModel(comRef)
            loco3dModel += [crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)]

        comInt = comRef.copy()
        """ go back down to average """
        for t in range(stepKnots):
            comRef[2] = comInit[2] -  t*comOffset/stepKnots
            dmodel = self.createFullSupportModel(comRef)
            loco3dModel += [crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)]
        
        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem

    
    def createOneFootLiftingProblem(self, x0, timeStep, supportKnots, stepKnots, footHeight):
        q0 = x0[:self.rmodel.nq]
        # q0[2] += self.ankle_offset 
        pin.forwardKinematics(self.rmodel, self.rdata, q0)
        pin.updateFramePlacements(self.rmodel, self.rdata)

        self.fpos0 = [self.rdata.oMf[fid].translation for fid in self.contact_ids]
        comRef = pin.centerOfMass(self.rmodel, self.rdata, q0) 

        flposInit = np.resize(self.rdata.oMf[self.flFootId].translation, 3)
        flposRef = np.resize(self.rdata.oMf[self.flFootId].translation, 3) 


        loco3dModel = []       
        for t in range(supportKnots):
            dmodel = self.createFullSupportModel(comRef)
            loco3dModel += [crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)]
        
        dx = .1/(stepKnots)
        "going up"
        upKnots = int(stepKnots/2)    
        for t in range(upKnots):
            flposRef[2] = flposInit[2] + t*footHeight/upKnots
            # print("desired foot initial height = %s"%flposInit[2])
            # print("desired foot height = %s"%flposRef[2])
            flposRef[0] = flposRef[0] + dx 
            # print("desired foot position = %s"%flposRef)
            dmodel = self.createSwingModel([self.frFootId, self.hlFootId, self.hrFootId], 
            [self.flFootId], comRef, [flposRef], [None], dampVelocity=None, trackWeight=1.e+2)
            loco3dModel += [crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)]
        "stay up " 
        # for t in range(3):
        #     flposRef[2] = flposInit[2] + footHeight
        #     # print("desired foot initial height = %s"%flposInit[2])
        #     # print("desired foot height = %s"%flposRef[2])
        #     flposRef[0] = flposRef[0] + dx 
        #     # print("desired foot position = %s"%flposRef)
        #     dmodel = self.createSwingModel([self.frFootId, self.hlFootId, self.hrFootId], 
        #     [self.flFootId], comRef, [flposRef], [None], dampVelocity=1.e-7, trackWeight=1.e+2)
        #     loco3dModel += [crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)]
        "going down"
        downKnots = stepKnots - upKnots
        for t in range(downKnots+1):
            flposRef[2] = flposInit[2] + footHeight * (1. - 1.*t/downKnots)   
            # print("desired foot initial height = %s"%flposInit[2])
            # print("desired foot height = %s"%flposRef[2])
            flposRef[0] = flposRef[0] + dx 
            # print("desired foot position = %s"%flposRef)
            if t>(downKnots-5): #  and self.WHICH_MEASUREMENT == "Contact":
                dV = 1.e-5 * t/downKnots
            else:
                dV = None 
            dmodel = self.createSwingModel([self.frFootId, self.hlFootId, self.hrFootId], 
            [self.flFootId], comRef, [flposRef], [None], dampVelocity=None, trackWeight=1.e+2)
            loco3dModel += [crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)]
        """ stabilize before activating contact ? """ 
        for t in range(5):
            # print("desired foot height = %s"%flposRef[2])
            dmodel = self.createSwingModel([self.frFootId, self.hlFootId, self.hrFootId], 
            [self.flFootId], comRef, [flposRef], [None], dampVelocity=1.e+2, trackWeight=1.e+2)
            loco3dModel += [crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)]
        """ go back to balance """ 
        for t in range(supportKnots):
            dmodel = self.createFullSupportModel(comRef)
            loco3dModel += [crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)]

        # supportIds = None 
        measurementModels = []
        """ Creating the Measurement Models """
        for t, model in enumerate(loco3dModel):
            if t in range(supportKnots):
                swingIds = []
                supportIds = [self.flFootId, self.frFootId, self.hlFootId, self.hrFootId]
            elif t in range(supportKnots, supportKnots + stepKnots+10):
                swingIds = [self.flFootId]
                supportIds = [self.frFootId, self.hlFootId, self.hrFootId] 
            else:
                swingIds = []
                supportIds = [self.flFootId, self.frFootId, self.hlFootId, self.hrFootId]  


            if self.WHICH_MEASUREMENT is None:
                pass 
            elif self.WHICH_MEASUREMENT == "Uniform":
                state_diffusion = .01 * np.eye(dmodel.state.ndx)
                state_noise = 1.e-7 * np.eye(dmodel.state.ndx)
                measurement_diffusion = .01 * np.eye(dmodel.state.ndx)
                measurement_noise = 5.e-3 * np.eye(dmodel.state.ndx) 
                swingQ_noise = [[0., 0., 0.] for _ in swingIds]
                swingdQ_noise = [[0., 0., 0.] for _ in swingIds]
                measurementMod = measurement.MeasurementModelSwingJoints(model,state_diffusion, 
                            state_noise, measurement_diffusion, measurement_noise, self.contact_names, swingIds, swingQ_noise, swingdQ_noise)

            elif self.WHICH_MEASUREMENT == "SwingJoints":
                state_diffusion = .01 * np.eye(dmodel.state.ndx)
                state_noise = 1.e-7 * np.eye(dmodel.state.ndx)
                measurement_diffusion = .01 * np.eye(dmodel.state.ndx)
                measurement_noise = 5.e-3 * np.eye(dmodel.state.ndx) 
                if t<(supportKnots+upKnots):
                    swingIds = None 
                    measurementMod = measurement.MeasurementModelSwingJoints(model,state_diffusion, 
                            state_noise, measurement_diffusion, measurement_noise, self.contact_names, swingIds)
                else:
                    swingQ_noise = [[5.e-2, 5.e-2, 5.e-2] for _ in swingIds]
                    swingdQ_noise = [[1.e-2, 1.e-2, 1.e-2] for _ in swingIds]
                    measurementMod = measurement.MeasurementModelSwingJoints(model,state_diffusion, 
                            state_noise, measurement_diffusion, measurement_noise, self.contact_names, swingIds,swingQ_noise, swingdQ_noise)
            elif self.WHICH_MEASUREMENT == "Unconstrained":
                state_diffusion = .01 * np.eye(model.state.ndx)
                state_noise = 1.e-7 * np.eye(model.state.ndx)
                measurement_diffusion = .01 * np.eye(model.state.ndx)
                measurement_noise = 1.e-3 * np.eye(model.state.ndx) 
                if t<(supportKnots+upKnots):
                    swingIds = None 
                    measurementMod = measurement.MeasurementModelSwingJoints(model,state_diffusion, 
                            state_noise, measurement_diffusion, measurement_noise, self.contact_names, swingIds)
                else:
                    swingQ_noise = [[5.e-1, 5.e-1, 5.e-1] for _ in swingIds]
                    swingdQ_noise = [[1.e-1, 1.e-1, 1.e-1] for _ in swingIds]
                    measurementMod = measurement.MeasurementModelContactNoise(model,state_diffusion, 
                            state_noise, measurement_diffusion, measurement_noise, swingIds ,swingQ_noise, swingdQ_noise)
            elif self.WHICH_MEASUREMENT == "Contact":
                if t<(supportKnots+upKnots):
                    state_diffusion = .01 * np.eye(model.state.ndx)
                    state_noise = 1.e-7 * np.eye(model.state.ndx)
                    measurement_diffusion = .01 * np.eye(model.state.ndx)
                    measurement_noise = 1.e-3 * np.eye(model.state.ndx) 
                    swingIds = None 
                    measurementMod = measurement.MeasurementModelSwingJoints(model,state_diffusion, 
                            state_noise, measurement_diffusion, measurement_noise, self.contact_names, swingIds)
                else:
                    state_diffusion = .01 * np.eye(model.state.ndx)
                    state_noise = 1.e-7 * np.eye(model.state.ndx)
                    measurement_diffusion = .01 * np.eye(model.state.ndx)
                    measurement_noise = 1.e-3 * np.eye(model.state.ndx) 
                    swingQ_noise = [[5.e-1, 5.e-1, 5.e-1] for _ in swingIds]
                    swingdQ_noise = [[1.e-1, 1.e-1, 1.e-1] for _ in swingIds]
                    measurementMod = measurement.MeasurementModelContactConsistent(model,state_diffusion, 
                            state_noise, measurement_diffusion, measurement_noise, swingIds, supportIds ,swingQ_noise, swingdQ_noise)

            else:
                raise BaseException("Measurement Model Not Recognized")

            measurementModels += [measurementMod]


        # problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return loco3dModel, measurementModels


    def createFullSupportModel(self, comTask=None):
        # create the cost and add com cost 
        costModel = crocoddyl.CostModelSum(self.state, self.actuation.nu)
        if comTask is not None:
            comTrack = crocoddyl.CostModelCoMPosition(self.state, comTask, self.actuation.nu)
            costModel.addCost("comTrack", comTrack, 1.e+3)
        # add the contact model and the friction cone cost  
        contactModel = crocoddyl.ContactModelMultiple(self.state, self.actuation.nu)
        for i, frame_id in enumerate(self.contact_ids):
            # xref = crocoddyl.FrameTranslation(frame_id, self.fpos0[i])
            xref = crocoddyl.FrameTranslation(frame_id, np.array([0., 0., 0.]))
            supportContactModel = crocoddyl.ContactModel3D(self.state, xref, self.actuation.nu, 
                                                            self.baumgarte)
            contactModel.addContact(self.rmodel.frames[frame_id].name + "_contact", 
                                    supportContactModel)
            # friction cone  
            cone = crocoddyl.FrictionCone(self.nsurf, self.mu, 4, False)
            frameCone = crocoddyl.FrameFrictionCone(frame_id, cone)
            frictionCone = crocoddyl.CostModelContactFrictionCone(
                self.state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub)),
                frameCone , self.actuation.nu)
            costModel.addCost(self.rmodel.frames[frame_id].name + "_frictionCone", frictionCone, 1.e+1)
            motion_ref = pin.Motion.Zero()
            vref = crocoddyl.FrameMotion(frame_id, motion_ref)
            FootVelCost = crocoddyl.CostModelFrameVelocity(self.state, vref, self.actuation.nu)
            costModel.addCost(self.rmodel.frames[frame_id].name + "_Vel", FootVelCost, 1.e-1)
        # state and control regulation 
        stateWeights = np.array([1.e-6] * 3 + [1.e+3] * 3 + [1.e-6] * 3 + [1.e+0] * 3 + [1.e+0] * 3 + [1.e+0] * 3 + [1.e+0] * 6 
                                + [1.e-3] * 3 + [1.e+0] * 3 + [1.e+0] * 3 + [1.e+0] * 3 )
        stateReg = crocoddyl.CostModelState(self.state,
                            crocoddyl.ActivationModelWeightedQuad(np.array(stateWeights**2)),
                            self.rmodel.defaultState, self.actuation.nu)
        ctrlReg = crocoddyl.CostModelControl(self.state, self.actuation.nu)
        costModel.addCost("stateReg", stateReg, 1.e-2)
        costModel.addCost("ctrlReg", ctrlReg, 1.e-5)
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, 
                                                                contactModel, costModel, 0., True)
        return dmodel 


    def createSwingModel(self, supportIds, swingIds, comTask=None, swingFeetTasks=None, 
                        swingVelocityTasks=[None], dampVelocity=None, trackWeight=1.e2): 
        costModel = crocoddyl.CostModelSum(self.state, self.actuation.nu)
        if comTask is not None:
            comTrack = crocoddyl.CostModelCoMPosition(self.state, comTask, self.actuation.nu)
            costModel.addCost("comTrack", comTrack, 1.e+3)
    #     # add the contact model and the friction cone cost  
        contactModel = crocoddyl.ContactModelMultiple(self.state, self.actuation.nu)
        for i, frame_id in enumerate(supportIds):
            # xref = crocoddyl.FrameTranslation(frame_id, self.fpos0[self.contact_ids.index(frame_id)])
            xref = crocoddyl.FrameTranslation(frame_id, np.array([0., 0., 0.]))
            supportContactModel = crocoddyl.ContactModel3D(self.state, xref, 
                                            self.actuation.nu, self.baumgarte)
            contactModel.addContact(self.rmodel.frames[frame_id].name + "_contact", supportContactModel)
            # friction cone  
            cone = crocoddyl.FrictionCone(self.nsurf, self.mu, 4, False)
            frameCone = crocoddyl.FrameFrictionCone(frame_id, cone)
            frictionCone = crocoddyl.CostModelContactFrictionCone(
                self.state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub)),
                frameCone , self.actuation.nu)
            costModel.addCost(self.rmodel.frames[frame_id].name + "_frictionCone", frictionCone, 1.e+1)
            motion_ref = pin.Motion.Zero()
            vref = crocoddyl.FrameMotion(frame_id, motion_ref)
            FootVelCost = crocoddyl.CostModelFrameVelocity(self.state, vref, self.actuation.nu)
            costModel.addCost(self.rmodel.frames[frame_id].name + "_Vel", FootVelCost, 1.e-1)
            # state and control regulation 
        for i, frame_id in enumerate(swingIds):
            xref = crocoddyl.FrameTranslation(frame_id,swingFeetTasks[i])
            footTrack = crocoddyl.CostModelFrameTranslation(self.state, xref, self.actuation.nu)
            costModel.addCost(self.rmodel.frames[frame_id].name + "_footTrack", footTrack, trackWeight) 
            # if swingVelocityTasks[i] is not None:
            #     impulseFootVelCost = crocoddyl.CostModelFrameVelocity(self.state, swingVelocityTasks[i], self.actuation.nu)
            #     costModel.addCost(self.rmodel.frames[frame_id].name + "_impulseVel", impulseFootVelCost, 1.e-5)
            if dampVelocity is not None:
                vref = crocoddyl.FrameMotion(frame_id, pin.Motion.Zero())
                impulseFootVelCost = crocoddyl.CostModelFrameVelocity(self.state, vref, self.actuation.nu)
                costModel.addCost(self.rmodel.frames[frame_id].name + "_impulseVel", impulseFootVelCost, dampVelocity)



        # 
        stateWeights = np.array([1.e-6] * 3 + [1.e+3] * 3 + [1.e-6] * 3 + [1.e+0] * 3 + [1.e+0] * 3 + [1.e+0] * 3 + [1.e+0] * 6 
                                + [1.e-3] * 3 + [1.e+0] * 3 + [1.e+0] * 3 + [1.e+0] * 3 )
        stateReg = crocoddyl.CostModelState(self.state,
                    crocoddyl.ActivationModelWeightedQuad(np.array(stateWeights**2)),
                    self.rmodel.defaultState, self.actuation.nu)
        ctrlReg = crocoddyl.CostModelControl(self.state, self.actuation.nu)
        costModel.addCost("stateReg", stateReg, 1.e-2)
        costModel.addCost("ctrlReg", ctrlReg, 1.e-5)
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, 
        self.actuation, contactModel, costModel, 0., True)
        return dmodel 
                                                         

    # def createSwitchModel(self, supportIds, swingIds, comTask=None, swingFeetTasks=None):
    #     costModel = crocoddyl.CostModelSum(self.state, self.actuation.nu)
    #     if comTask is not None:
    #         comTrack = crocoddyl.CostModelCoMPosition(self.state, comTask, self.actuation.nu)
    #         costModel.addCost("comTrack", comTrack, 1.e+2)
    # #     # add the contact model and the friction cone cost  
    #     contactModel = crocoddyl.ContactModelMultiple(self.state, self.actuation.nu)
    #     for i, frame_id in enumerate(supportIds):
    #         # xref = crocoddyl.FrameTranslation(frame_id, self.fpos0[self.contact_ids.index(frame_id)])
    #         xref = crocoddyl.FrameTranslation(frame_id, np.array([0., 0., 0.]))
    #         supportContactModel = crocoddyl.ContactModel3D(self.state, xref, 
    #                                         self.actuation.nu, self.baumgarte)
    #         contactModel.addContact(self.rmodel.frames[frame_id].name + "_contact", supportContactModel)
    #         # friction cone  
    #         cone = crocoddyl.FrictionCone(self.nsurf, self.mu, 4, False)
    #         frameCone = crocoddyl.FrameFrictionCone(frame_id, cone)
    #         frictionCone = crocoddyl.CostModelContactFrictionCone(
    #             self.state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub)),
    #             frameCone , self.actuation.nu)
    #         costModel.addCost(self.rmodel.frames[frame_id].name + "_frictionCone", frictionCone, 1.e-2)
    #         # state and control regulation 
    #     for i, frame_id in enumerate(swingIds):
    #         xref = crocoddyl.FrameTranslation(frame_id,swingFeetTasks[i])
    #         footTrack = crocoddyl.CostModelFrameTranslation(self.state, xref, self.actuation.nu)
    #         costModel.addCost(self.rmodel.frames[frame_id].name + "_footTrack", footTrack, 1.e-1) 
    #         vref = crocoddyl.FrameMotion(frame_id, pin.Motion.Zero())
    #         impulseFootVelCost = crocoddyl.CostModelFrameVelocity(self.state, vref, self.actuation.nu)
    #         costModel.addCost(self.rmodel.frames[frame_id].name + "_impulseVel", impulseFootVelCost, 1.e1)

    #     # 
    #     stateWeights = np.array([0.] * 3 + [50.] * 3 + [.01] * 3 + [10.] * 3 
    #                             + [10.] * 3 + [10.] * 3 + [10.] * 6 
    #                             + [1.] * (self.rmodel.nv - 6))
    #     stateReg = crocoddyl.CostModelState(self.state,
    #                 crocoddyl.ActivationModelWeightedQuad(np.array(stateWeights**2)),
    #                 self.rmodel.defaultState, self.actuation.nu)
    #     ctrlReg = crocoddyl.CostModelControl(self.state, self.actuation.nu)
    #     costModel.addCost("stateReg", stateReg, 1.e-3)
    #     costModel.addCost("ctrlReg", ctrlReg, 1.e-5)
    #     dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, 
    #     self.actuation, contactModel, costModel, 0., True) 
    #     return dmodel


    def log_reference(self, supportIds=None, swingIds=None, comTask=None, 
                        swingFeetTasks=None, swingVelocityTasks=None):
        pass 

    def interpolate_state(self, x1, x2, d):
        """ interpolate state for feedback at higher rate that plan """
        x = np.zeros(self.rmodel.nq+self.rmodel.nv)
        x[:self.rmodel.nq] =  pin.interpolate(self.rmodel, x1[:self.rmodel.nq], x2[:self.rmodel.nq], d)
        x[self.rmodel.nq:] = x1[self.rmodel.nq:] + d*(x2[self.rmodel.nq:] - x1[self.rmodel.nq:])
        return x