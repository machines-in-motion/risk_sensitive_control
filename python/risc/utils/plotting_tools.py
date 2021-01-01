import numpy as np 
import pinocchio as pin
import matplotlib.pyplot as plt 


class RobotPlottingTools(object):
    def __init__(self, robot, contact_names):
        """ robot_wrapper.RobotWrapper object """

        self.robot = robot 
        self.model = self.robot.model 
        self.data = self.model.createData()
        self.nq = self.model.nq 
        self.nv = self.model.nv 
        self.n = self.nq + self.nv 
        self.m = None 
        self.nx = 2*self.nv
        self.contact_names = contact_names
        self.contact_ids = [self.model.getFrameId(name) for name in self.contact_names]
        self.nc = len(contact_names)
        self.state_names = []
        self.control_names = []
        self.branch_names = []
        self.branch_joints = []
        self.branch_ids = []
        self.planar = False 
        self.parse_model()
        if self.planar:
            self.njoints = self.nv - 3 
            self.nq_base = 4 
            self.nv_base = 3 
        else:
            self.njoints = self.nv - 6 
            self.nq_base = 7 
            self.nv_base = 6
        
        self.sT = np.zeros([self.nv, self.m])
        self.sT[self.nv_base:, :] = np.eye(self.m)


    def parse_model(self):
        # look at base joint if planar or spatial 
        # extract base state 
        if self.model.joints[1].nq == 4:
            # planar base 
            joint_pos = ['x', 'z', 'qx', 'qz']
            joint_vel = ['vx', 'vz', 'wy']
            self.planar = True 
        elif self.model.joints[1].nq == 7:
            joint_pos = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
            joint_vel = ['vx', 'vy', 'vz', 'wx', 'wy', 'wz']
        else:
            raise NotImplementedError('root joint not recognized !')
        # loop over joints and construct states and controls 
        for n in self.model.names[2:]:
            joint_pos += [n]
            joint_vel += ['d'+n+'/dt']
            self.control_names += [n]
        assert len(joint_pos) == self.nq, 'nq size missmatch'
        assert len(joint_vel) == self.nv, 'nv size missmatch'
        self.state_names = joint_pos + joint_vel
        self.m = len(self.control_names)
        # define the branches to plot 
        self.branch_names = []
        root_id = self.model.getJointId('root_joint')
        # define the branches 
        for i, j in enumerate(self.model.joints):
            if self.model.parents[i] == root_id:
                self.branch_names += [['root_joint', self.model.names[i]]]
                self.branch_joints += [[self.model.names[i]]]
        # now skip the universe and root_joint 
        for i,j in enumerate(self.model.joints[2:]):
            for k, branch in enumerate(self.branch_names):
                if self.model.names[self.model.parents[j.id]] == branch[-1]:
                    self.branch_names[k] += [self.model.names[j.id]]
                    self.branch_joints[k] += [self.model.names[j.id]]
        # add the names of the cotact points 
        for contact in self.contact_names:
            for k, branch in enumerate(self.branch_names):
                contact_id = self.model.getFrameId(contact)
                if self.model.names[self.model.frames[contact_id].parent] in branch:
                    self.branch_names[k] += [contact]
        # collect the branch ids 
        self.branch_ids = [[self.model.getFrameId(name) for name in branch] 
                                            for branch in self.branch_names]

    def plotting_points(self, x):
        pin.framesForwardKinematics(self.model, self.data, x[:self.nq, None])
        com = pin.centerOfMass(self.model, self.data, x[:self.nq, None])
        com = np.resize(com, 3)
        positions = []
        for branch in self.branch_ids:
            brnch = []
            for frm in branch:
                brnch += [list(np.resize(self.data.oMf[frm].
                                         translation, 3))]
            positions += [brnch]
        xpos = [[f[0] for f in b] for b in positions]
        ypos = [[f[1] for f in b] for b in positions]
        zpos = [[f[2] for f in b] for b in positions]
        return positions, xpos, ypos, zpos, com
    
    def sagittal_plt(self, x):
        """ plots x,z plane """
        coordinates, xpos, ypos, zpos, com = self.plotting_points(x)
        plt.figure('Sagittal Plane', figsize=(8, 8))
        for branch in coordinates:
            i = coordinates.index(branch)
            plt.plot(xpos[i], zpos[i], 'o-', linewidth=2.5,
                     label=self.branch_names[i][1][:2])
        plt.scatter(com[0], com[2],linewidth=10., label='CoM')
        plt.legend()
        plt.xlabel('X position')
        plt.ylabel('Z position')
        plt.axes().set_aspect('equal', 'datalim')
        plt.title('Sagittal Plane')
        plt.grid(True)

    def sagittal_trajectory(self, xt, grid=True):
        colors = ['bo-', 'go-', 'ro-','ko-']
        N = len(xt)
        plt.figure('robot', figsize=(10, 10))
        for i in range(N):
            state = xt[i]
            coordinates, xpos, ypos, zpos, cm \
                = self.plotting_points(state)
            opacity = i / (1. * N)
            if opacity <= .3:
                opacity = .3
            for branch in coordinates:
                j = coordinates.index(branch)
                plt.plot(xpos[j], zpos[j], colors[j], linewidth=2.5,
                         alpha=opacity)
            plt.scatter(cm[0], cm[2], linewidth=10.)
        # last one only separated so it shows legend only once
        state = xt[-1]

        coordinates, xpos, ypos, zpos, cm \
            = self.plotting_points(state)
        opacity = i / (1. * N)
        for branch in coordinates:
            j = coordinates.index(branch)
            plt.plot(xpos[j], zpos[j], colors[j], linewidth=2.5,
                     alpha=opacity, label=self.branch_names[j][1][:2])
        plt.scatter(cm[0], cm[2], linewidth=7., label='CoM')

        plt.xlabel('X position')
        plt.ylabel('Z position')
        plt.legend()
        plt.axes().set_aspect('equal', 'datalim')
        plt.title('Sagittal Plane')
        if grid:
            plt.grid(True)

    def base_position(self):
        pass 

    def simulator_contact_forces(self, ft, dt=1.e-3):
        fig, ax = plt.subplots(3, 1, figsize=(8, 8))
        # convert ft to ndarray 
        forces = np.zeros([len(ft), 3*self.nc])

        for t in range(len(ft)):
            forces[t,:] = np.resize(ft[t],3*self.nc)

        time_array = dt * np.arange(len(ft))
        direction = ['fx_', 'fy_', 'fz_']
        for i,d in enumerate(direction):
            for k, n in enumerate(self.contact_names):
                ax[i].plot(time_array, forces[:,3*k+i], linewidth=1.2, label=d+n)
            ax[i].grid()
            ax[i].legend()

    def contact_positions(self, xt, dt=1.e-3):
        fig, ax = plt.subplots(3, 1, figsize=(8, 8))
        positions = np.zeros([len(xt), 3*self.nc])
        for t, xi in enumerate(xt):
            pin.framesForwardKinematics(self.model, self.data, np.resize(xi[:self.nq],(self.nq,1)))
            for i, ind in enumerate(self.contact_ids):
                positions[t,3*i:3*i+3] = np.resize(self.data.oMf[ind].translation,3)
        time_array = dt*np.arange(len(xt))
        direction = ['_x', '_y', '_z']
        for i, d in enumerate(direction):
            for k, n in enumerate(self.contact_names):
                ax[i].plot(time_array, positions[:, 3*k+i],
                           linewidth=1.2, label=n+d)
            ax[i].grid()
            ax[i].legend()

    def get_solver_forces_positions(self, solver):
        """ loops over entire trajectory, logs contact forces in global 
        frame and contact positions """
        contact_forces = np.zeros(
            (len(solver.xs[:-1]), 3*len(self.contact_names)))
        contact_positions = np.zeros(
            (len(solver.xs[:-1]), 3*len(self.contact_names)))
        for i, d in enumerate(solver.problem.runningDatas):
            pin.framesForwardKinematics(
                self.model, self.data, solver.xs[i][:self.nq])
            m = solver.problem.runningModels[i]
            for k, c_key in enumerate(self.contact_names):
                c_id = self.model.getFrameId(c_key)
                omf = self.data.oMf[c_id]
                contact_positions[i, 3*k:3*k +
                                       3] = np.resize(omf.translation, 3)
                try:
                    c_data = d.differential.multibody.contacts.contacts[c_key+'_contact']
                    contact_forces[i, 3*k:3*k+3] = np.resize(c_data.jMf.actInv(c_data.f).linear, 3)
                except:
                    pass
        return contact_forces, contact_positions



    def solver_contact_forces(self, solver, dt=1.e-2): 
        forces, positions = self.get_solver_forces_positions(solver)
        time_array = dt * np.arange(len(solver.us))
        fig, ax = plt.subplots(2, 1, figsize=(8, 8))
        for i, cnt in enumerate(self.contact_names):
            ax[0].plot(time_array, forces[:, 3*i+2],
                       linewidth=1.2, label=cnt+'_fz')
            ax[1].plot(time_array,
                       np.sqrt(forces[:, 3*i]**2 + forces[:, 3*i+1]**2),
                       linewidth=1.2, label=cnt+'_ft')

        # ax[0].grid()
        ax[0].legend()
        # ax[1].grid()
        ax[1].legend()


    def solver_frame_positions(self, frame_names, solver, dt=1.e-2):
        time_array = dt * np.arange(len(solver.xs))
        N = time_array.shape[0]
        n = len(frame_names) 
        position = np.zeros([N, n, 3])
        frame_ids = [self.model.getFrameId(fn) for fn in frame_names]

        for t in range(N): 
            pin.framesForwardKinematics(self.model, self.data, solver.xs[t][:self.nq])
            for i in range(n):
                position[t,i, :] = np.resize(self.data.oMf[frame_ids[i]].translation, 3)


        direction = ['_x',  '_y', '_z']
        colors = ['r', 'g', 'b', 'm', 'c', 'y', 'k']
        fig, ax = plt.subplots(3,1)
        for i in range(3):
            for j in range(n):
                ax[i].plot(time_array, position[:,j,i], colors[j], label= frame_names[j]+direction[i])
            ax[i].legend()
        fig.canvas.set_window_title('Frame Positions')
        return [fig]


    def joint_impedance(self, solvers, dt=1.e-2):
        # plots feedback from all states to every particular actuator
        time_array = 1.e-2 * np.arange(len(solvers[0].us))
        Knorm_solvers = []
        Bnorm_solvers = []
        nv_base = 3 if self.planar else 6
        for solver in solvers:
            k_solver = []
            b_solver = []
            for i, cn in enumerate(self.control_names):
                p_joint = [np.linalg.norm(fb[i, :self.nv]) for fb in solver.K]
                d_joint = [np.linalg.norm(fb[i, self.nv:]) for fb in solver.K]
                k_solver += [p_joint]
                b_solver += [d_joint]

            Knorm_solvers += [k_solver]
            Bnorm_solvers += [b_solver]
        # plot for joint 
        for i, cn in enumerate(self.control_names):
            fig, ax = plt.subplots(2,1,figsize=(8,10))
            for j in range(len(solvers)):
                ax[0].plot(time_array, Knorm_solvers[j][i], label='solver %s'%j)
                ax[1].plot(time_array, Bnorm_solvers[j][i], label='solver %s'%j)
            ax[0].legend()
            ax[0].grid()
            ax[0].title.set_text(cn+' stiffness')
            ax[1].legend()
            ax[1].grid()
            ax[1].title.set_text(cn+' damping')

    def diagonal_impedance(self, solvers, dt=1.e-2):
        # only the diagonal elements, a joint and its relative actuator 
        time_array = 1.e-2 * np.arange(len(solvers[0].us))
        Knorm_solvers = []
        Bnorm_solvers = []
        nv_base = 3 if self.planar else 6
        for solver in solvers:
            k_solver = []
            b_solver = []
            for i, cn in enumerate(self.control_names):
                p_joint = [np.linalg.norm(fb[i, nv_base+i]) for fb in solver.K]
                d_joint = [np.linalg.norm(fb[i, self.nv+nv_base+i])
                           for fb in solver.K]
                k_solver += [p_joint]
                b_solver += [d_joint]

            Knorm_solvers += [k_solver]
            Bnorm_solvers += [b_solver]
        # plot for joint
        for i, cn in enumerate(self.control_names):
            fig, ax = plt.subplots(2, 1, figsize=(8, 10))
            for j in range(len(solvers)):
                ax[0].plot(time_array, Knorm_solvers[j]
                           [i], label='solver %s' % j)
                ax[1].plot(time_array, Bnorm_solvers[j]
                           [i], label='solver %s' % j)
            ax[0].legend()
            ax[0].grid()
            ax[0].title.set_text(cn+' stiffness')
            ax[1].legend()
            ax[1].grid()
            ax[1].title.set_text(cn+' damping')



    def branch_impedances(self, solvers, dt=1.e-2):
        time_array = 1.e-2 * np.arange(len(solvers[0]))

        Knorm_solvers = []
        Bnorm_solvers = []
        nv_base = 3 if self.planar else 6

        for solver in solvers:
            k_solver = []
            b_solver = []

            for i, bj in enumerate(self.branch_joints):
                branch_start = self.control_names.index(bj[0])
                branch_end = 1+self.control_names.index(bj[-1])
                p_branch = [np.linalg.norm(fb[branch_start:branch_end, :self.nv])
                            for fb in solver]
                d_branch = [np.linalg.norm(
                    fb[branch_start:branch_end, self.nv:]) for fb in solver]
                k_solver += [p_branch]
                b_solver += [d_branch]

            Knorm_solvers += [k_solver]
            Bnorm_solvers += [b_solver]
        figs = []
        for i, bj in enumerate(self.branch_joints):
            fig, ax = plt.subplots(2, 1, figsize=(8, 10))
            for j in range(len(solvers)):
                ax[0].plot(time_array, Knorm_solvers[j][i],
                        linewidth=1.2, label='solver %s' % j)
                ax[1].plot(time_array, Bnorm_solvers[j][i],
                        linewidth=1.2, label='solver %s' % j)
            fig.canvas.set_window_title(bj[0][:2]+' branch impedance')
            ax[0].legend()
            ax[0].grid()
            ax[0].title.set_text(bj[0][:2]+' branch stiffness')
            ax[1].legend()
            ax[1].grid()
            ax[1].title.set_text(bj[0][:2]+' branch damping')
            figs += [fig]
        return figs


    def risk_sensitive_filter_cov(self, solver, dt=1.e-2):
        try:
            time_array = dt*np.arange(len(solver.filterGains))
        except:
            raise BaseException('not a risk sensitive solver')
        
        gain_norms = []
        cov_norms = []

        for i,k in enumerate(solver.filterGains):
            gain_norms += [np.linalg.norm(k)]
            cov_norms += [np.linalg.norm(solver.cov[i])]
        fig, ax = plt.subplots(2,1, figsize=(8,8))
        ax[0].plot(time_array, gain_norms,
                   linewidth=2., label='filter norms')
        ax[1].plot(time_array, cov_norms,
                   linewidth=2., label='covariance norm')
        ax[0].legend()
        ax[0].grid()
        ax[0].title.set_text('filter norms')
        ax[1].legend()
        ax[1].grid()
        ax[1].title.set_text('covariance norms')
        return [fig]



    def leg_controls(self, solver, dt=1.e-2):
        time_array = dt * np.arange(len(solver.us))
        us = np.zeros([len(solver.us), self.m])
        for t, ui in enumerate(solver.us):
            us[t] = np.resize(ui, len(self.control_names))
        
        fig, ax = plt.subplots(len(self.branch_names), 1, figsize=(12, 12))
        bind = 0
        for i,b in enumerate(self.branch_names):
            for j, c in enumerate(b[1:-1]):
                ax[i].plot(time_array, us[:,bind+j], linewidth=1.2, label=c)
            ax[i].grid()
            ax[i].legend()
            bind += len(b[1:-1])
        return [fig]

    
    def com_tracking(self, solver, xt, dt=1.e-2): 
        assert len(xt)==len(solver.xs), 'trajectories not of equal dimensions'
        time_array = dt * np.arange(len(solver.xs))

        com_ref = np.zeros([len(xt),3])
        com_act = np.zeros([len(xt),3]) 

        for t in range(len(xt)):
            pin.centerOfMass(self.model, self.data, solver.xs[t][:self.nq])
            com_ref[t] = np.resize(self.data.com, 3)
            pin.centerOfMass(self.model, self.data, xt[t][:self.nq, None])
            com_act[t] = np.resize(self.data.com, 3)

        direction = ['x', 'y', 'z']
        fig, ax = plt.subplots(3,1)
        for i in range(3): 
            ax[i].plot(time_array, com_ref[:,i], 'r', linewidth=1.2, label=direction[i]+' ref')
            ax[i].plot(time_array, com_act[:,i], 'g', linewidth=1.2, label=direction[i]+' act')
            ax[i].legend()
            ax[i].grid()
        
        fig.canvas.set_window_title('com tracking')
        return [fig] 

    def joint_tracking(self, solver, xt, dt=1.e-2):
        """takes a solver and a simulator trajectory """
        assert len(xt)==len(solver.xs), 'trajectories not of equal dimensions'
        time_array = dt * np.arange(len(solver.xs))
        joint_ref = np.zeros([len(xt), self.njoints])
        joint_act = np.zeros([len(xt), self.njoints]) 

        for t in range(len(xt)):
            joint_ref[t] = np.resize(solver.xs[t][self.nq_base:self.nq], self.njoints)
            joint_act[t] = np.resize(xt[t][self.nq_base:self.nq], self.njoints)
        
        fig, ax = plt.subplots(self.njoints, 1)
        for i in range(self.njoints): 
            ax[i].plot(time_array, joint_ref[:,i], 'r', linewidth=1.2, label=self.state_names[self.nq_base+i]+' ref')
            ax[i].plot(time_array, joint_act[:,i], 'g', linewidth=1.2, label=self.state_names[self.nq_base+i]+' act')
            ax[i].legend()
            ax[i].grid()
        
        fig.canvas.set_window_title('joint tracking')
        return [fig] 

    def contact_position_tracking(self, solver, xt, dt=1.e-2):
        """ plot contact position tracking in x,y,z """
        assert len(xt)==len(solver.xs), 'trajectories not of equal dimensions'
        time_array = dt * np.arange(len(solver.xs))

        contact_pos_ref = np.zeros([len(xt),len(self.contact_names),3])
        contact_pos_act = np.zeros([len(xt),len(self.contact_names),3])

        for t in range(len(xt)):
            pin.framesForwardKinematics(self.model, self.data, solver.xs[t][:self.nq])
            for i, cid in enumerate(self.contact_ids):
                contact_pos_ref[t,i,:] = np.resize(self.data.oMf[cid].translation, 3)
            pin.framesForwardKinematics(self.model, self.data, xt[t][:self.nq, None])
            for i, cid in enumerate(self.contact_ids):
                contact_pos_act[t,i,:] = np.resize(self.data.oMf[cid].translation, 3)

        figs = []
        direction = ['x','y', 'z']
        for j, cn in enumerate(self.contact_names):
            fig, ax = plt.subplots(3, 1)
            for i in range(3):
                ax[i].plot(time_array, contact_pos_ref[:,j,i], 'r', linewidth=1.2, label=cn[:2]+' ref '+ direction[i])
                ax[i].plot(time_array, contact_pos_act[:,j,i], 'g', linewidth=1.2, label=cn[:2]+' act '+ direction[i])
                ax[i].legend()
                # ax[i].grid()
            fig.canvas.set_window_title(cn[:2] + ' position tracking')
            figs += [fig]
        return figs

    def contact_velocity(self, xt, dt=1.e-2):

        time_array = dt * np.arange(len(xt))
        contact_vel = np.zeros([len(xt),len(self.contact_names),3])
        for t in range(len(xt)):
            pin.framesForwardKinematics(self.model, self.data, xt[t][:self.nq])
            pin.computeJointJacobians(self.model, self.data, xt[t][:self.nq])
            for i, cid in enumerate(self.contact_ids):
                jac = pin.getFrameJacobian(self.model, self.data, cid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3,:]
                contact_vel[t,i,:] = np.resize(jac.dot(xt[t][self.nq:]), 3)
        figs = []
        direction = ['dx','dy', 'dz']
        for j, cn in enumerate(self.contact_names):
            fig, ax = plt.subplots(3, 1)
            for i in range(3):
                ax[i].plot(time_array, contact_vel[:,j,i], 'b', linewidth=1.2, label=cn[:2]+ direction[i])
                ax[i].legend()
                ax[i].grid()
            fig.canvas.set_window_title(cn[:2] + ' velocity')
            figs += [fig]
        return figs


    def contact_velocity_tracking(self, solver, xt, dt=1.e-2):
        """ plot contact velocity tracking in x,y,z """
        assert len(xt)==len(solver.xs), 'trajectories not of equal dimensions'
        time_array = dt * np.arange(len(solver.xs))

        contact_vel_ref = np.zeros([len(xt),len(self.contact_names),3])
        contact_vel_act = np.zeros([len(xt),len(self.contact_names),3])

        for t in range(len(xt)):
            pin.framesForwardKinematics(self.model, self.data, solver.xs[t][:self.nq])
            pin.computeJointJacobians(self.model, self.data, solver.xs[t][:self.nq])
            for i, cid in enumerate(self.contact_ids):
                jac = pin.getFrameJacobian(self.model, self.data, cid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3,:]
                contact_vel_ref[t,i,:] = np.resize(jac.dot(solver.xs[t][self.nq:]), 3)
            pin.framesForwardKinematics(self.model, self.data, xt[t][:self.nq, None])
            pin.computeJointJacobians(self.model, self.data, xt[t][:self.nq, None])
            for i, cid in enumerate(self.contact_ids):
                jac = pin.getFrameJacobian(self.model, self.data, cid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3,:]
                contact_vel_act[t,i,:] = np.resize(jac.dot(xt[t][self.nq:,None]), 3)

        figs = []
        direction = ['dx','dy', 'dz']
        for j, cn in enumerate(self.contact_names):
            fig, ax = plt.subplots(3, 1)
            for i in range(3):
                ax[i].plot(time_array, contact_vel_ref[:,j,i], 'r', linewidth=1.2, label=cn[:2]+' ref '+ direction[i])
                ax[i].plot(time_array, contact_vel_act[:,j,i], 'g', linewidth=1.2, label=cn[:2]+' act '+ direction[i])
                ax[i].legend()
                ax[i].grid()
            fig.canvas.set_window_title(cn[:2] + ' velocity tracking')
            figs += [fig]
        return figs


    def base_tracking(self, solver, xt, dt=1.e-2):
        # 4 figs, base position, base orientation, base angular velocity 
        assert len(xt)==len(solver.xs), 'trajectories not of equal dimensions'
        time_array = dt * np.arange(len(solver.xs))
        #
        base_pos_ref, base_pos_act = np.zeros([len(xt), 3]), np.zeros([len(xt), 3])     
        base_ori_ref, base_ori_act = np.zeros([len(xt), 4]), np.zeros([len(xt), 4])
        base_lin_ref, base_lin_act = np.zeros([len(xt), 3]), np.zeros([len(xt), 3])
        base_ang_ref, base_ang_act = np.zeros([len(xt), 3]), np.zeros([len(xt), 3])
        #
        for t in range(len(xt)):
            base_pos_ref[t], base_ori_ref[t] = np.resize(solver.xs[t][:3],3), np.resize(solver.xs[t][3:7],4)
            base_lin_ref[t], base_ang_ref[t] = np.resize(solver.xs[t][self.nq:self.nq+3],3), np.resize(solver.xs[t][self.nq+3:self.nq+6],3)
            base_pos_act[t], base_ori_act[t] = np.resize(xt[t][:3],3), np.resize(xt[t][3:7],4) 
            base_lin_act[t], base_ang_act[t] = np.resize(xt[t][self.nq+3:self.nq+6],3), np.resize(xt[t][self.nq+3:self.nq+6],3)
        #
        figs = []
        #
        fig, ax = plt.subplots(2,1)
        for i in range(3):
            ax[0].plot(time_array,base_pos_ref[:,i],'--', label=self.state_names[i]+' ref')
            ax[0].plot(time_array,base_pos_act[:,i], label=self.state_names[i]+' act')
            ax[1].plot(time_array,base_ori_ref[:,i],'--', label=self.state_names[i+3]+' ref')
            ax[1].plot(time_array,base_ori_act[:,i], label=self.state_names[i+3]+' act')
        ax[1].plot(time_array,base_ori_ref[:,3],'--', label=self.state_names[6]+' ref')
        ax[1].plot(time_array,base_ori_act[:,3], label=self.state_names[6]+' act')
        ax[0].legend()
        # ax[0].grid()
        ax[1].legend()
        # ax[1].grid()
        #
        fig.canvas.set_window_title('base pose tracking')
        figs += [fig]
        # 
        fig, ax = plt.subplots(2,1)
        for i in range(3):
            ax[0].plot(time_array,base_lin_ref[:,i],'--', label=self.state_names[self.nq+i]+' ref')
            ax[0].plot(time_array,base_lin_act[:,i], label=self.state_names[self.nq+i]+' act')
            ax[1].plot(time_array,base_ang_ref[:,i],'--', label=self.state_names[self.nq+i+3]+' ref')
            ax[1].plot(time_array,base_ang_act[:,i], label=self.state_names[self.nq+i+3]+' act')
        ax[0].legend()
        # ax[0].grid()
        ax[1].legend()
        # ax[1].grid()
        fig.canvas.set_window_title('base velocity tracking')
        figs += [fig]
        return figs 

    
    def force_tracking(self, solver, ft, dt=1.e-2):
        time_array_act = dt*np.arange(len(ft))
        solver_forces, _ = self.get_solver_forces_positions(solver)
        time_array_ref = dt * np.arange(solver_forces.shape[0])
        figs = []
        directions = ['x', 'y', 'z']
        sim_forces = np.zeros([len(ft), 3*self.nc])

        for t in range(len(ft)):
            sim_forces[t,:] = np.resize(ft[t],3*self.nc)

        for i, cn in enumerate(self.contact_names):
            fig, ax = plt.subplots(3,1)
            for j in range(3):
                if j ==0:
                    ax[j].plot(time_array_ref, -solver_forces[:,3*i+j] ,'r--',label=cn[:2]+' f_'+directions[j]+' ref')
                else:
                    ax[j].plot(time_array_ref, solver_forces[:,3*i+j] ,'r--',label=cn[:2]+' f_'+directions[j]+' ref')
                ax[j].plot(time_array_act, sim_forces[:,3*i+j], 'g', label=cn[:2]+' f_'+directions[j]+' act')
                ax[j].legend()
                # ax[j].grid()
            fig.canvas.set_window_title(cn[:2]+' force tracking')
            figs += [fig]

        return figs 

    def control_tracking(self, solver, ut, dt=1.e-2):
        N = len(ut)
        time_array = dt*np.arange(N) 
        u_ref = np.zeros([N,self.m])
        u_act = np.zeros([N,self.m])
        for t in range(N):
            u_ref[t] = np.resize(solver.us[t], self.m)
            u_act[t] = np.resize(ut[t], self.m)
        
        fig,ax = plt.subplots(self.m, 1)
        for i, un in enumerate(self.control_names):
            ax[i].plot(time_array, u_ref[:,i], 'r--', label=un+' ref')
            ax[i].plot(time_array, u_act[:,i], 'g', label=un+' act')
            ax[i].legend()
            # ax[i].grid()
        fig.canvas.set_window_title('actual control tracking')
        return fig 

    def compare_frame_impedace(self, feedback, xt, frame_name, dt=1.e-2):
        N = len(feedback[0])
        t_array = dt*np.arange(N)
        fid = self.model.getFrameId(frame_name)
        # t, solver, Kp or Kd, dir(x,y,z)
        impedance = np.zeros([N,len(feedback), 2, 3])
        for t in range(N):
            pin.computeJointJacobians(self.model, self.data, xt[t][:self.nq, None])
            pin.framesForwardKinematics(self.model, self.data, xt[t][:self.nq, None])
            j = pin.getFrameJacobian(self.model, self.data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            for i in range(len(feedback)):
                impedance[t,i,0,0] = np.linalg.norm(j[:3,6:].dot(feedback[i][t][:,:self.nv])[0,:])
                impedance[t,i,0,1] = np.linalg.norm(j[:3,6:].dot(feedback[i][t][:,:self.nv])[1,:])
                impedance[t,i,0,2] = np.linalg.norm(j[:3,6:].dot(feedback[i][t][:,:self.nv])[2,:])
                impedance[t,i,1,0] = np.linalg.norm(j[:3,6:].dot(feedback[i][t][:,self.nv:])[0,:])
                impedance[t,i,1,1] = np.linalg.norm(j[:3,6:].dot(feedback[i][t][:,self.nv:])[1,:])
                impedance[t,i,1,2] = np.linalg.norm(j[:3,6:].dot(feedback[i][t][:,self.nv:])[2,:])
        # 
        figs = []
        direction = ['x', 'y', 'z']
        fig, ax = plt.subplots(3,1)
        for i in range(3):
            for j in range(len(feedback)):
                ax[i].plot(t_array, impedance[:,j,0,i], label=direction[i]+' kp '+'solver %s'%j) 
            # ax[i].grid()
            ax[i].legend()
        fig.canvas.set_window_title(frame_name + ' stiffness')
        figs += [fig]
        fig, ax = plt.subplots(3,1)
        for i in range(3):
            for j in range(len(feedback)):
                ax[i].plot(t_array, impedance[:,j,1,i], label=direction[i]+' kd '+'solver %s'%j) 
            # ax[i].grid()
            ax[i].legend()
        fig.canvas.set_window_title(frame_name + ' damping')
        figs += [fig]
        # 
        return figs 

    
    def compare_feedforward(self, solvers, dt=1.e-2):
        N = len(solvers[0])
        t_array = dt*np.arange(N)
        controls = np.zeros([N,len(solvers), self.m])
        for t in range(N):
            for i in range(len(solvers)):
                controls[t,i,:] = np.resize(solvers[i][t], self.m)

        fig,ax = plt.subplots(self.m, 1)
        for i, un in enumerate(self.control_names):
            for j in range(len(solvers)):
                ax[i].plot(t_array, controls[:,j,i], label=un + ' solver %s'%j)
            ax[i].legend()
            # ax[i].grid()

        return fig 

    def get_reference_forces(self, solver, dt=1.e-2):
        solver_forces, _ = self.get_solver_forces_positions(solver)
        return solver_forces


    def compare_simulation_froces(self, solver, sim_forces, dt=1.e-2, names=['DDP', 'Risk']): 
        t_array = dt*np.arange(len(sim_forces[0]))
        solver_forces, _ = self.get_solver_forces_positions(solver)
        forces = np.zeros([t_array.shape[0], len(sim_forces), 3*self.nc])
        for t in range(t_array.shape[0]): 
            for i, simf in enumerate(sim_forces):
                forces[t,i, :] = np.resize(simf[t],3*self.nc)
        
        figs = []
        direction = [' $f_x$ ', ' $f_y$ ', ' $f_z$ ']
        colors = ['b', 'g', 'r', 'm', 'k']
        # colors = ['b', 'g-', 'r--', 'm-.', 'k:']
        # colors = ['b', 'k:', 'r--', 'm-.', 'g-']
        # colors = ['b', 'r--', 'c:', 'm-.', 'g-']
        for i,d in enumerate(direction):
            fig, ax = plt.subplots(2,1, figsize=(15,15))
            for j in range(2):
                if i == 0:
                    ax[j].plot(t_array[:-1], -1.*solver_forces[:,3*j+i], '--k',linewidth=4.,  label='ref')
                else:
                    ax[j].plot(t_array[:-1], solver_forces[:,3*j+i], '--k',linewidth=4.,  label='ref')
                for k, name in enumerate(names):
                    ax[j].plot(t_array, forces[:,k,3*j+i], colors[k], linewidth=4.,  label=name)
                if j == 0:
                    ax[j].legend(loc='upper left')
                ax[j].set_xlabel('time [s]')
                ax[j].set_ylabel(self.contact_names[j][:2] +d+  ' [N]')
                # if j == 0:
                #    ax[j].set_ylim([-1., 12.])
                # ax[j].set_ylim([0., 20.])
                # ax[j].grid() 
            fig.canvas.set_window_title('f_'+d + ' comparison')
            fig.tight_layout()
            figs += [fig]
        return figs

    def compare_simulation_controls(self, sim_controls, dt=1.e-2, names=['DDP', 'Risk']):
        t_array = dt*np.arange(len(sim_controls[0]))
        controls = np.zeros([t_array.shape[0], len(sim_controls), self.m])

        for t in range(t_array.shape[0]):
            for i, simu in enumerate(sim_controls):
                controls[t,i,:] = np.resize(simu[t], self.m)

        figs = []
        colors = ['b', 'g', '--r']
        fig, ax = plt.subplots(self.m, 1)
        for i, cn in enumerate(self.control_names):
            for j, sn in enumerate(names):
                ax[i].plot(t_array, controls[:,j,i], colors[j], label=cn +' '+sn)
            ax[i].legend(loc='upper right')
            # ax[i].grid()
        fig.canvas.set_window_title('control input comparison')
        figs += [fig]
        return figs 

    def contact_space_impedance(self, frame_name, solver_imp, states, dt=1.e-2):
        t_array = dt*np.arange(len(solver_imp[0]))
        kp = np.zeros([t_array.shape[0], len(solver_imp), self.m, self.nv])
        kd = np.zeros([t_array.shape[0], len(solver_imp), self.m, self.nv])
        P = np.zeros([t_array.shape[0], len(solver_imp),3,3])
        D = np.zeros([t_array.shape[0], len(solver_imp),3,3])

        frame_id = self.model.getFrameId(frame_name)
        for t in range(t_array.shape[0]):
            for i in range(len(solver_imp)):
                xt = states[i]
                pin.computeJointJacobians(self.model, self.data, xt[t][:self.nq, None])
                pin.framesForwardKinematics(self.model, self.data, xt[t][:self.nq, None])
                jac =  pin.getFrameJacobian(self.model, self.data, frame_id, 
                                            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3,:]
                pin.computeMinverse(self.model, self.data, xt[t][:self.nq, None])
                jminv = jac.dot(self.data.Minv)
                mcontact = np.linalg.inv(jminv.dot(jac.T))
                jpsd = jac.T.dot(np.linalg.inv(jac.dot(jac.T)))
                kp[t,i,:,:] = solver_imp[i][t][:,:self.nv]
                kd[t,i,:,:] = solver_imp[i][t][:,self.nv:]
                P[t,i,:,:] = mcontact.dot(jminv).dot(self.sT).dot(kp[t,i,:,:]).dot(jpsd)
                D[t,i,:,:] = mcontact.dot(jminv).dot(self.sT).dot(kd[t,i,:,:]).dot(jpsd)

        figs = []
        solver_names = [' DDP ', ' Risk ']
        for k in range(len(solver_imp)):
            fig, ax = plt.subplots(2,1)
            for i in range(3):
                for j in range(3):
                    ax[0].plot(t_array, P[:,k,i,j])
                    ax[1].plot(t_array, D[:,k,i,j])
            fig.canvas.set_window_title(frame_name + ' Stiffness and Damping for %s'%solver_names[k])
            figs += [fig]
        return figs
    
    def compare_frame_position(self, frame_name, sim_xs, ref_xs, dt=1.e-2):
        t_array = dt*np.arange(len(ref_xs))
        N = t_array.shape[0]
        n = len(sim_xs) + 1
        position = np.zeros([N, n, 3])
        frame_id = self.model.getFrameId(frame_name)

        for t in range(N): 
            pin.framesForwardKinematics(self.model, self.data, ref_xs[t][:self.nq])
            position[t,-1,:] = np.resize(self.data.oMf[frame_id].translation, 3)
            for i in range(len(sim_xs)):
                pin.framesForwardKinematics(self.model, self.data, sim_xs[i][t][:self.nq,None])
                position[t,i,:] = np.resize(self.data.oMf[frame_id].translation, 3)

        direction = ['x',  'y', 'z']
        solver_name = [' DDP ', ' Risk ', ' Ref ']
        colors = ['b', 'g', '--r']
        fig, ax = plt.subplots(3,1)
        for i in range(3):
            for j in range(n):
                ax[i].plot(t_array, position[:,j,i], colors[j], label= solver_name[j]+direction[i])
            ax[i].legend()
            ax[i].set_ylim([-.02, .14])
        
        
        fig.canvas.set_window_title(frame_name + ' position tracking')
        return [fig]

    def compare_base_tracking(self, sim_states, xref, dt=1.e-2, names= [' DDP ', ' Risk ', ' Ref ']):
        t_array = dt * np.arange(len(xref))
        N = t_array.shape[0]
        n = len(sim_states) #+1

        position = np.zeros([N, n, 3])

        for t in range(N):
            # position[t,-1,:] = np.resize(xref[t][:3],3)
            for i in range(len(sim_states)):
                position[t,i,:] = np.resize(sim_states[i][t][:3],3)

        
        direction = ['x',  'y', 'z']
        # solver_name = [' DDP ', ' Risk ', ' Ref ']
        colors = ['b', 'g-', 'r--', 'm-.', 'k:']
        fig, ax = plt.subplots(2,1, figsize=(15,15))
        for i in range(1,3):
            for j in range(n):
                ax[i-1].plot(t_array, position[:,j,i], colors[j], linewidth=4.,  label= names[j])
            ax[i-1].legend(loc='upper left')
            ax[i-1].set_xlabel('time [s]')
            ax[i-1].set_ylabel('base '+ direction[i] + ' [m]')
            
        fig.canvas.set_window_title('Base position tracking')
        fig.tight_layout()
        return [fig]

    def plot_frame_positions(self, xt, frame_name, dt=1.e-2):
        """ Plots xyz locations for a certain frame along a given trajectory """
        
        t_array = dt*np.arange(len(xt))
        N = t_array.shape[0]
        position = np.zeros([N,3])
        direction = ['x',  'y', 'z']
        frame_id = self.model.getFrameId(frame_name)
        # 
        for t in range(N): 
            pin.framesForwardKinematics(self.model, self.data, xt[t][:self.nq, None])
            position[t,:] = np.resize(self.data.oMf[frame_id].translation, 3)
        # 
        fig, ax = plt.subplots(3,1)
        for i in range(3):
            ax[i].plot(t_array, position[:,i], label= frame_name + ' ' + direction[i])
        fig.canvas.set_window_title(frame_name + ' position')
        return fig
       
                

        

    def compare_frame_height(self, frame_names, sim_xs, ref_xs, dt=1.e-2, names=[' DDP', ' Risk', ' ref'], setlim=True):
        t_array = dt*np.arange(len(ref_xs))
        N = t_array.shape[0]
        n = len(sim_xs)  #+ 1
        position = np.zeros([N, n, 4])
        frame_ids = [self.model.getFrameId(fn) for fn in frame_names]
        # 
        for k, fid in enumerate(frame_ids): 
            for t in range(N): 
                pin.framesForwardKinematics(self.model, self.data, ref_xs[t][:self.nq])
                position[t,-1,k] = self.data.oMf[fid].translation[2]
                for i in range(len(sim_xs)):
                    pin.framesForwardKinematics(self.model, self.data, sim_xs[i][t][:self.nq])
                    position[t,i,k] = self.data.oMf[fid].translation[2]
        # 
        colors = ['b', 'g-', 'r--', 'm-.', 'k:']
        fig, ax = plt.subplots(2,1, figsize=(15,15))
        for i in range(2):
            for j in range(n):
                ax[i].plot(t_array, position[:,j,i], colors[j], linewidth=4., label= names[j])
            ax[i].legend(loc='upper left')
            ax[i].set_xlabel('time [s]')
            ax[i].set_ylabel(frame_names[i][:2] + ' height [m]')
            if setlim:
                ax[i].set_ylim([0., .16])
        # 
        fig.canvas.set_window_title('feet height')
        fig.tight_layout()
        return [fig]

