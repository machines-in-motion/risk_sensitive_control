""" here we define measurement model and data """

""" 
A general discrete time model looks like 

x_{t+1} = f(x_t, u_t) + F(x_t, u_t) w_t 
y_{t+1} = h(x_t, u_t) + H(x_t, u_t) g_t


the local linear approximations then become 

x_{t+1} = A_t x_t + B_t u_t + C_t w_t 
y_{t+1} = F_t x_t + E_t u_t + D_t g_t

now for us, we are assuming full state measurement, i.e y = x 

the only difference will occur at C_t, D_t, w_t, g_t   
"""
import numpy as np 
import pinocchio as pin




class MeasurementModels(object):
    def __init__(self, running_models, measurement_models):
        self.rmodels = running_models
        self.measurementModels = measurement_models
        self.runningDatas = []

        for t, mModel in enumerate(self.measurementModels):
            self.runningDatas += [mModel.createData()]

#_____________________________________________________________________________________________________________________#

class MeasurementModelFullState(object):
    def __init__(self, model, sd, sn, md, mn):
        """ Defines a uniform measurement model, i.e. identity for the diffusion 
        nx: dimension of state vector x 
        ndx: dimension of tangent space at x ( = nx -1 when base orientation is included)
        ny: measurement dimension 
        sd: state diffusion 
        sn: state noise 
        md: measurement diffusion 
        mn: measurement noise 
         """

        self.model = model 
        self.nx, self.ndx, self.nu = model.state.nx, model.state.ndx, model.nu
        self.ny = self.ndx 
        self.sd = sd
        self.sn = sn
        self.md = md
        self.mn = mn  
        self.measurement = np.zeros(self.ny)
        self.MeasurementDataType = MeasurementDataFullState
        

    def createData(self):
        return self.MeasurementDataType(self) 

    def calc(self, data, x, u=None): 
        self.measurement[:] = data.xnext.copy()  

    def calcDiff(self, data, mdata, x , u=None, recalc=True):
        if recalc:
            self.calc(data, x, u)
        mdata.dx[:,:] = data.Fx.copy() 
        mdata.du[:,:] =  data.Fu.copy()

    def processNoise(self):
        """ return a sample noise vector """
        raise NotImplementedError('processNoise Not Implemented')

    def measurementNoise(self):
        """ return a sample noise vector """
        raise NotImplementedError('processNoise Not Implemented')


class MeasurementDataFullState(object): 
    def __init__(self, model):
        self.dx = np.zeros([model.ndx, model.ndx])
        self.du = np.zeros([model.ndx, model.nu])
        self.cc = model.sd.dot(model.sn).dot(model.sd.T)
        self.dd = model.md.dot(model.mn).dot(model.md.T)
        self.covMatrix = np.zeros([model.ndx, model.ndx])
#_____________________________________________________________________________________________________________________#

class MeasurementModelSwingJoints(object):
    def __init__(self, model, sd, sn, md, mn, contactNames, swingIds=None, swingPosNoise=None, swingVelNoise=None):
        """ A Full State Measurement Model with Noise Added Diagonally on joints 
            corresponding to the branch of SwingId """
        self.model = model
        self.pin_model = model.state.pinocchio
        self.pin_data = self.pin_model.createData()
        self.nx, self.ndx, self.nu = model.state.nx, model.state.ndx, model.nu
        self.nq = self.pin_model.nq 
        self.nv = self.pin_model.nv 
        self.ny = self.ndx
        self.sd = sd
        self.sn = sn
        self.md = md
        self.mn = mn
        self.measurement = np.zeros(self.nx)
        self.MeasurementDataType = MeasurementDataFullState
        self.contact_names = contactNames
        self.contact_ids = [self.pin_model.getFrameId(name) for name in self.contact_names]
        self.nc = len(contactNames)
        self.state_names = []
        self.control_names = []
        self.branch_names = []
        self.branch_joints = []
        self.branch_ids = []
        self.parse_model()
        self.njoints = self.nv - 6 
        self.nq_base = 7 
        self.nv_base = 6
        self.swingIds = swingIds
        self.swingPosNoise = swingPosNoise
        self.swingVelNoise = swingVelNoise
        if self.swingIds is not None: 
            assert len(self.swingIds) == len(self.swingPosNoise), "swingPosNoise Dimension Missmatch"
            assert len(self.swingIds) == len(self.swingVelNoise), "swingVelNoise Dimension Missmatch"
        # find active branches
        self.active_branches = []
        self.q_indices = []
        self.dq_indices = []

        if self.swingIds is not None:
            for fid in self.swingIds:
                for i, branch in enumerate(self.branch_ids):
                    if fid in branch:
                        self.active_branches += [i]
            # now collect state indeces 
            
            for i in self.active_branches:
                q_inds = [self.state_names.index(jn) - 1 for jn in self.branch_joints[i]]
                dq_inds = [self.nv-1+self.state_names.index(jn) for jn in self.branch_joints[i]]
                self.q_indices += [q_inds]
                self.dq_indices += [dq_inds]

    def createData(self):
        return self.MeasurementDataType(self)

    def joint_noise(self):
        """ checks active branches and adds nosie to thier corresponding joints """
        jn = np.zeros([self.ndx, self.ndx])
        if self.swingIds is not None:
            for i in range(len(self.swingIds)):
                for j, qind in enumerate(self.q_indices[i]):
                    jn[qind,qind] = self.swingPosNoise[i][j]
                    jn[self.dq_indices[i][j],self.dq_indices[i][j]] = self.swingVelNoise[i][j]
        return jn 

    def calc(self, data, mdata, x, u=None):
        self.measurement[:] = data.xnext.copy()
        # 
        mn = self.mn.copy()
        if self.swingIds is not None:
            mn += self.joint_noise()  
        mdata.covMatrix[:,:] = mn.copy()
        pn = self.sn.copy()
        mdata.dd = self.md.dot(mn).dot(self.md.T)
        mdata.cc = self.sd.dot(pn).dot(self.sd.T)

    def calcDiff(self, data, mdata, x, u=None, recalc=True):
        # if recalc:
        #     self.calc(data, mdata, x, u)
        self.calc(data, mdata, x, u)
        mdata.dx[:,:] = data.Fx.copy() 
        mdata.du[:,:] =  data.Fu.copy()

    def parse_model(self):
        # look at base joint if planar or spatial 
        # extract base state 
        if self.pin_model.joints[1].nq == 4:
            # planar base 
            joint_pos = ['x', 'z', 'qx', 'qz']
            joint_vel = ['vx', 'vz', 'wy']
            self.planar = True 
        elif self.pin_model.joints[1].nq == 7:
            joint_pos = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
            joint_vel = ['vx', 'vy', 'vz', 'wx', 'wy', 'wz']
        else:
            raise NotImplementedError('root joint not recognized !')
        # loop over joints and construct states and controls 
        for n in self.pin_model.names[2:]:
            joint_pos += [n]
            joint_vel += ['d'+n+'/dt']
            self.control_names += [n]
        assert len(joint_pos) == self.nq, 'nq size missmatch'
        assert len(joint_vel) == self.nv, 'nv size missmatch'
        self.state_names = joint_pos + joint_vel
        self.m = len(self.control_names)
        # define the branches to plot 
        self.branch_names = []
        root_id = self.pin_model.getJointId('root_joint')
        # define the branches 
        for i, j in enumerate(self.pin_model.joints):
            if self.pin_model.parents[i] == root_id:
                self.branch_names += [['root_joint', self.pin_model.names[i]]]
                self.branch_joints += [[self.pin_model.names[i]]]
        # now skip the universe and root_joint 
        for i,j in enumerate(self.pin_model.joints[2:]):
            for k, branch in enumerate(self.branch_names):
                if self.pin_model.names[self.pin_model.parents[j.id]] == branch[-1]:
                    self.branch_names[k] += [self.pin_model.names[j.id]]
                    self.branch_joints[k] += [self.pin_model.names[j.id]]
        # add the names of the cotact points 
        for contact in self.contact_names:
            for k, branch in enumerate(self.branch_names):
                contact_id = self.pin_model.getFrameId(contact)
                if self.pin_model.names[self.pin_model.frames[contact_id].parent] in branch:
                    self.branch_names[k] += [contact]
        # collect the branch ids 
        self.branch_ids = [[self.pin_model.getFrameId(name) for name in branch] 
                                            for branch in self.branch_names]


#_____________________________________________________________________________________________________________________# 

class MeasurementModelContactNoise(object):
    def __init__(self, model, sd, sn, md, mn, swingIds, swingPosNoise, swingVelNoise):
        self.model = model
        self.pin_model = model.state.pinocchio
        self.pin_data = self.pin_model.createData()
        self.nx, self.ndx, self.nu = model.state.nx, model.state.ndx, model.nu
        self.nq = self.pin_model.nq 
        self.nv = self.pin_model.nv 
        self.ny = self.ndx
        self.sd = sd
        self.sn = sn
        self.md = md
        self.mn = mn
        self.measurement = np.zeros(self.nx)
        self.MeasurementDataType = MeasurementDataContactNoise
        self.swingIds = swingIds
        self.swingPosNoise = swingPosNoise
        self.swingVelNoise = swingVelNoise
        if self.swingIds is not None: 
            assert len(self.swingIds) == len(self.swingPosNoise), "swingPosNoise Dimension Missmatch"
            assert len(self.swingIds) == len(self.swingVelNoise), "swingVelNoise Dimension Missmatch"
            # print ' measurement model created with swing Ids ', self.swingIds
        
    def frame_jacobian_derivative(self, fid): 
        """ returns frame jacobian and its time derivative expressed in local frame """
        
        j = pin.getFrameJacobian(self.pin_model, self.pin_data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        djdt = pin.getFrameJacobianTimeVariation(self.pin_model, self.pin_data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return j ,djdt 

    def full_state_noise(self, x):
        jn = np.zeros([self.ndx, self.ndx])
        if self.swingIds is not None:
            pin.forwardKinematics(self.pin_model, self.pin_data, 
                                    x[:self.pin_model.nq], x[self.pin_model.nq:])
            pin.computeJointJacobians(self.pin_model, self.pin_data, x[:self.pin_model.nq])
            pin.computeJointJacobiansTimeVariation(self.pin_model, self.pin_data, 
                                    x[:self.pin_model.nq], x[self.pin_model.nq:])
            pin.updateFramePlacements(self.pin_model, self.pin_data)
            # pin.framesForwardKinematics(self.pin_model, self.pin_data, x[:self.pin_model.nq])
            for i, fid in enumerate(self.swingIds):
                jac, djac = self.frame_jacobian_derivative(fid)
                transform = np.zeros([6, self.ndx]) # maps joints errors to ee errors in pos and vel 
                transform[:3,:self.nv] = jac[:3,:].copy()
                transform[3:,self.nv:] = jac[:3,:].copy()
                transform[3:,:self.nv] = djac[:3,:].copy()
                # get pseudo inverse 
                invTransform = transform.T.dot(np.linalg.inv(transform.dot(transform.T)))
                cov_local = np.diag(self.swingPosNoise[i] + self.swingVelNoise[i])
                jn += invTransform.dot(cov_local).dot(invTransform.T)
        return jn 
        
        
        

    def createData(self):
        return self.MeasurementDataType(self)

    def calc(self, data, mdata, x, u=None):
        self.measurement[:] = data.xnext.copy()
        # 
        mn = self.mn.copy()
        if self.swingIds is not None:
            mn += self.full_state_noise(x)  
        mdata.covMatrix[:,:] = mn.copy()  
        pn = self.sn.copy()
        mdata.dd = self.md.dot(mn).dot(self.md.T)
        mdata.cc = self.sd.dot(pn).dot(self.sd.T)



    def calcDiff(self, data, mdata, x, u=None, recalc=True):
        if recalc:
            pass
            # self.calc(data, mdata, x, u)
        self.calc(data, mdata, x, u)
        # mdata.dx[:,:] = np.eye(self.ndx)
        mdata.dx[:, :] = data.Fx.copy()
        mdata.du[:, :] = data.Fu.copy()

    def processNoise(self):
        """ return a sample noise vector """
        raise NotImplementedError('processNoise Not Implemented')

    def measurementNoise(self):
        """ return a sample noise vector """
        raise NotImplementedError('processNoise Not Implemented')


class MeasurementDataContactNoise(object):
    def __init__(self, model):
        self.dx = np.zeros([model.ndx, model.ndx])
        self.du = np.zeros([model.ndx, model.nu])
        self.cc = model.sd.dot(model.sn).dot(model.sd.T)
        self.dd = model.md.dot(model.mn).dot(model.md.T)
        self.covMatrix = np.zeros([model.ndx, model.ndx])



#_____________________________________________________________________________________________________________________# 

class MeasurementModelContactConsistent(object):
    def __init__(self, model, sd, sn, md, mn, swingIds, supportIds, swingPosNoise, swingVelNoise):
        self.model = model
        self.pin_model = model.state.pinocchio
        self.pin_data = self.pin_model.createData()
        self.nx, self.ndx, self.nu = model.state.nx, model.state.ndx, model.nu
        self.nq = self.pin_model.nq 
        self.nv = self.pin_model.nv 
        self.ny = self.ndx
        self.sd = sd
        self.sn = sn
        self.md = md
        self.mn = mn
        self.measurement = np.zeros(self.nx)
        self.MeasurementDataType = MeasurementDataContactConsistent
        self.swingIds = swingIds
        self.supportIds = supportIds    
        self.swingPosNoise = swingPosNoise
        self.swingVelNoise = swingVelNoise
        if self.swingIds is not None: 
            assert len(self.swingIds) == len(self.swingPosNoise), "swingPosNoise Dimension Missmatch"
            assert len(self.swingIds) == len(self.swingVelNoise), "swingVelNoise Dimension Missmatch"
            # print ' measurement model created with swing Ids ', self.swingIds
        
    def frame_jacobian_derivative(self, fid): 
        """ returns frame jacobian and its time derivative expressed in local frame """
        
        j = pin.getFrameJacobian(self.pin_model, self.pin_data, fid, 
                                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3,:]
        djdt = pin.getFrameJacobianTimeVariation(self.pin_model, self.pin_data, fid, 
                                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3,:]
        return j ,djdt 

    def full_state_noise(self, x):
        jn = np.zeros([self.ndx, self.ndx])
        if self.swingIds is not None:
            for i, fid in enumerate(self.swingIds):
                jac, djac = self.frame_jacobian_derivative(fid)
                transform = np.zeros([6, self.ndx]) # maps joints errors to ee errors in pos and vel 
                transform[:3,:self.nv] = jac.copy()
                transform[3:,self.nv:] = jac.copy()
                transform[3:,:self.nv] = djac.copy()
                # get pseudo inverse 
                psdInvTransform = transform.T.dot(np.linalg.inv(transform.dot(transform.T)))
                cov_local = np.diag(self.swingPosNoise[i] + self.swingVelNoise[i])
                jn += psdInvTransform.dot(cov_local).dot(psdInvTransform.T)
        return jn 
    
    def null_space_projector(self):
        P = np.eye(self.ndx)
        if self.supportIds is not None:
            jc = np.zeros([3*len(self.supportIds), self.nv])
            djc =  np.zeros([3*len(self.supportIds), self.nv])
            Ac = np.zeros([6*len(self.supportIds), self.ndx])
            for i, fid in enumerate(self.supportIds):
                jc[3*i:3*i+3, :] = pin.getFrameJacobian(self.pin_model, self.pin_data, fid, 
                                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3,:]
                djc[3*i:3*i+3, :] = pin.getFrameJacobianTimeVariation(self.pin_model, self.pin_data, fid, 
                                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3,:]
            Ac[:3*len(self.supportIds), :self.nv]= jc.copy()
            Ac[3*len(self.supportIds):, self.nv:]= jc.copy()
            Ac[3*len(self.supportIds):, :self.nv]= djc.copy()
            # 
            psdA = Ac.T.dot(np.linalg.inv(Ac.dot(Ac.T)))
            P -= psdA.dot(Ac)
        return P 
            
    def createData(self):
        return self.MeasurementDataType(self)

    def calc(self, data, mdata, x, u=None):
        self.measurement[:] = data.xnext.copy()
        mn = self.mn.copy()
        pin.forwardKinematics(self.pin_model, self.pin_data, 
                                    x[:self.pin_model.nq], x[self.pin_model.nq:])
        pin.computeJointJacobians(self.pin_model, self.pin_data, x[:self.pin_model.nq])
        pin.computeJointJacobiansTimeVariation(self.pin_model, self.pin_data, 
                                x[:self.pin_model.nq], x[self.pin_model.nq:])
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        # pin.framesForwardKinematics(self.pin_model, self.pin_data, x[:self.pin_model.nq])
        if self.swingIds is not None:
            if self.supportIds is not None:
                Pc = self.null_space_projector()
                assert Pc.shape == (self.ndx, self.ndx), 'Null Space Projector Has the Wrong Dimension'
                mn += Pc.dot(self.full_state_noise(x)).dot(Pc.T)   
            else:
                mn += self.full_state_noise(x)   
        # 
        mdata.covMatrix[:,:] = mn.copy()
        pn = self.sn.copy()
        mdata.dd = self.md.dot(mn).dot(self.md.T)
        mdata.cc = self.sd.dot(pn).dot(self.sd.T)

    def calcDiff(self, data, mdata, x, u=None, recalc=True):
        if recalc:
            pass
            # self.calc(data, mdata, x, u)
        self.calc(data, mdata, x, u)
        mdata.dx[:,:] = np.eye(self.ndx)
        # mdata.dx[:, :] = data.Fx.copy()
        # mdata.du[:, :] = data.Fu.copy()

    def processNoise(self):
        """ return a sample noise vector """
        raise NotImplementedError('processNoise Not Implemented')

    def measurementNoise(self):
        """ return a sample noise vector """
        raise NotImplementedError('processNoise Not Implemented')


class MeasurementDataContactConsistent(object):
    def __init__(self, model):
        self.dx = np.zeros([model.ndx, model.ndx])
        self.du = np.zeros([model.ndx, model.nu])
        self.cc = model.sd.dot(model.sn).dot(model.sd.T)
        self.dd = model.md.dot(model.mn).dot(model.md.T)
        self.covMatrix = np.zeros([model.ndx, model.ndx])
