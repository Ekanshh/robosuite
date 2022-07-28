from controller.base_controller import Controller
from copy import deepcopy
import utils.transform_utils as T
from utils.control_utils import *
from scipy.linalg import expm
import matplotlib.pyplot as plt
import pandas as pd


class ImpedanceController(Controller):
    def __init__(self,
                 observation,
                 init_data,
                 action,
                 model_timestep=0.002,
                 joint_dim=6,
                 control_dim=108,
                 input_max=1,
                 input_min=-1,
                 output_max=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
                 output_min=(-0.05, -0.05, -0.05, -0.5, -0.5, -0.5),
                 kp=3000,
                 damping_ratio=1,
                 kp_limits=(0, 100000),
                 damping_ratio_limits=(0, 10),
                 uncouple_pos_ori=True,
                 method='rotation'
                 ):
        super().__init__(
            observation,
            model_timestep,
            method,
            joint_dim
        )

        self.control_dim = control_dim

        # input and output max and min (allow for either explicit lists or single numbers)
        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)
        self.output_max = self.nums2array(output_max, self.control_dim)
        self.output_min = self.nums2array(output_min, self.control_dim)

        # kp and kd limits
        self.kp_min = self.nums2array(kp_limits[0], 6)
        self.kp_max = self.nums2array(kp_limits[1], 6)
        self.damping_ratio_min = self.nums2array(damping_ratio_limits[0], 6)
        self.damping_ratio_max = self.nums2array(damping_ratio_limits[1], 6)

        # clip kp and kd
        kp = np.clip(kp, self.kp_min, self.kp_max)

        # kp kd
        self.kp = self.nums2array(kp, 6)
        self.kd = 2 * np.sqrt(self.kp) * damping_ratio
        # print(f"self.kp: {self.kp}")
        # print(f"self.kd: {self.kd}")
        # self.kd[3:] = [3, 3, 3]
        self.kp_impedance = deepcopy(self.kp)
        self.kd_impedance = deepcopy(self.kd)

        # whether or not pos and ori want to be uncoupled
        self.uncoupling = uncouple_pos_ori

        # initialize goals based on initial pos / ori
        #   TODO: set goal to the final pos+ori
        self.goal_ori = np.array(self.initial_ee_ori_mat)
        self.goal_pos = np.array(self.initial_ee_pos)
        self.goal_vel = np.array(self.ee_pos_vel)
        self.goal_ori_vel = np.array(self.initial_ee_ori_vel)
        self.impedance_vec = np.zeros(12)
        self.switch = 1
        self.enter = 0
        self.force_filter = np.zeros((3, 1))

        self.relative_ori = np.zeros(3)
        self.ori_ref = None
        self.set_desired_goal = False
        self.desired_pos = np.zeros(12)
        self.torques = np.zeros(6)
        self.F0 = np.zeros(6)
        self.F0[2] = 0
        self.F_int = np.zeros(6)
        self.bias = 1
        # ee resets - bias at initial state
        self.ee_sensor_bias = 0

        self.peg_name = init_data.peg.name
        self.hole_name = init_data.hole.name

        self.set_control_param(action)

        # for graphs
        self.ee_pos_vec_x, self.ee_pos_vec_y, self.ee_pos_vec_z = [], [], []
        self.impedance_model_pos_vec_x, self.impedance_model_pos_vec_y, self.impedance_model_pos_vec_z = [], [], []
        self.time_vec, self.error_pos_vec = [], []
        self.ee_ori_vec_x, self.ee_ori_vec_y, self.ee_ori_vec_z = [], [], []
        self.impedance_ori_vec_x, self.impedance_ori_vec_y, self.impedance_ori_vec_z = [], [], []
        self.desired_torque_x, self.desired_torque_y, self.desired_torque_z = [], [], []
        self.desired_force_x, self.desired_force_y, self.desired_force_z = [], [], []
        self.ee_sensor_bias_x, self.ee_sensor_bias_y, self.ee_sensor_bias_z = [], [], []
        self.Fx_no_bias_sensor_frame, self.Fy_no_bias_sensor_frame, self.Fz_no_bias_sensor_frame = [], [], []
        self.Fx_no_bias, self.Fy_no_bias, self.Fz_no_bias = [], [], []
        self.wernce_vec_int_Fx, self.wernce_vec_int_Fy, self.wernce_vec_int_Fz = [], [], []
        self.wernce_vec_int_Mx, self.wernce_vec_int_My, self.wernce_vec_int_Mz = [], [], []
        self.pos_min_jerk_x, self.pos_min_jerk_y, self.pos_min_jerk_z = [], [], []
        self.ori_min_jerk_x, self.ori_min_jerk_y, self.ori_min_jerk_z = [], [], []
        self.eef_wernce_vec_int_Fx, self.eef_wernce_vec_int_Fy, self.eef_wernce_vec_int_Fz = [], [], []
        self.eef_wernce_vec_int_Mx, self.eef_wernce_vec_int_My, self.eef_wernce_vec_int_Mz = [], [], []
        self.real_force_x, self.real_force_y, self.real_force_z = [], [], []
        self.ori_vel_min_jerk_x, self.ori_vel_min_jerk_y, self.ori_vel_min_jerk_z = [], [], []
        self.ee_ori_vel_vec_x, self.ee_ori_vel_vec_y, self.ee_ori_vel_vec_z = [], [], []
        self.ori_vel_min_jerk_orig_x, self.ori_vel_min_jerk_orig_y, self.ori_vel_min_jerk_orig_z = [], [], []
        self.impedance_ori_vel_vec_x, self.impedance_ori_vel_vec_y, self.impedance_ori_vel_vec_z = [], [], []
        self.impedance_vel_vec_x, self.impedance_vel_vec_y, self.impedance_vel_vec_z = [], [], []
        self.vel_min_jerk_x, self.vel_min_jerk_y, self.vel_min_jerk_z = [], [], []
        self.ee_vel_vec_x, self.ee_vel_vec_y, self.ee_vel_vec_z = [], [], []

    def set_goal(self,
                 observation
                 ):
        """
        Sets goal based on input @action. If self.impedance_mode is not "fixed", then the input will be parsed into the
        delta values to update the goal position / pose and the kp and/or damping_ratio values to be immediately updated
        internally before executing the proceeding control loop.

        Note that @action expected to be in the following format, based on impedance mode!

            :param observation:
            :param joint_vel:
            :param joint_pos:
            :param ee_ori_vel:
            :param ee_ori_mat:
            :param ee_pos_vel:
            :param ee_pos:
            :Mode `'fixed'`: [joint pos command]
            :Mode `'variable'`: [damping_ratio values, kp values, joint pos command]
            :Mode `'variable_kp'`: [kp values, joint pos command]

        Args:
            action (Iterable): Desired relative joint position goal state
            set_pos (Iterable): If set, overrides @action and sets the desired absolute eef position goal state
            set_ori (Iterable): IF set, overrides @action and sets the desired absolute eef orientation goal state
        """
        # Update state
        self.update(observation)

        if self.switch == 1:
            if self.time - self.t_bias < self.t_final:
                self.goal_pos, self.goal_ori, self.goal_vel, self.goal_ori_vel = self.built_next_desired_point()
            else:
                self.goal_pos, self.goal_ori, self.goal_vel, self.goal_ori_vel = self.desired_vec_fin[-1]
        else:
            self.goal_pos, self.goal_ori, self.goal_vel, self.goal_ori_vel = self.built_next_desired_point()
        #   set the desire_vec=goal
        self.set_desired_goal = True

    def run_controller(self, ignore_impedance: bool = False):
        """
        Calculates the torques required to reach the desired setpoint.

        Impedance Position Base (IM-PB) -- position and orientation.

        work in world space

        Returns:
             np.array: Command torques
        """
        # Update state
        # self.update(observation)
        # self.kp = deepcopy(np.clip(self.kp_impedance, self.kp_min, self.kp_max))
        # self.kd = deepcopy(np.clip(self.kd_impedance, 0.0, 4 * 2 * np.sqrt(self.kp) * np.sqrt(2)))

        self.desired_pos = np.concatenate((self.goal_pos, self.goal_ori, self.goal_vel, self.goal_ori_vel), axis=0)

        if self.method == 'euler':
            self.desired_pos[3:6] = deepcopy(orientation_error(T.euler2mat(self.desired_pos[3:6]), self.ee_ori_mat))
            ori_error = self.desired_pos[3:6]
            ori_min_jerk = deepcopy(self.desired_pos[3:6])
            ori_real = T.Rotation_Matrix_To_Vector(self.final_orientation, self.ee_ori_mat)
            # goal_ori = R.from_euler('zyx', self.goal_ori, degrees=False).as_rotvec()
            self.desired_pos[9:12] = deepcopy(self.euler2WorldAngle_vel(self.desired_pos[9:12]))
            ori_vel_min_jerk = deepcopy(self.desired_pos[9:12])

        if self.bias:
            self.ee_sensor_bias = deepcopy(np.concatenate(
                (self.ee_ori_mat @ -self.sensor_force, self.ee_ori_mat @ -self.sensor_torque), axis=0))
            if self.time > 0.00001:
                # print(f"self.kd: {self.kd}")
                # print(f"self.kp: {self.kp}")
                self.bias = 0
            # print(f"self.ee_sensor_bias: {self.ee_sensor_bias}")
        if self.has_contacts() and self.enter == 0:
            self.enter = 1

        self.F_int_no_bias_sensor_frame = (np.concatenate((self.sensor_force, self.sensor_torque), axis=0))

        self.F_int_no_bias = (np.concatenate(
            (self.ee_ori_mat @ -self.sensor_force, self.ee_ori_mat @ -self.sensor_torque),
            axis=0))

        self.F_int = (np.concatenate(
            (self.ee_ori_mat @ -self.sensor_force, self.ee_ori_mat @ -self.sensor_torque),
            axis=0) - self.ee_sensor_bias)
        if self.enter:
            self.kp = deepcopy(np.clip(self.kp_impedance, self.kp_min, self.kp_max))
            self.kd = deepcopy(np.clip(self.kd_impedance, 0.0, 4 * 2 * np.sqrt(self.kp) * np.sqrt(2)))

            if not ignore_impedance:
                if self.bias == 0:
                    print("Using impedance")
                    self.impedance_vec = deepcopy(self.desired_pos)
                    self.bias = None

                self.F_int = (np.concatenate(
                    (self.ee_ori_mat @ -self.sensor_force, self.ee_ori_mat @ -self.sensor_torque),
                    axis=0) - self.ee_sensor_bias)

                self.desired_pos = deepcopy(
                    self.ImpedanceEq(self.F_int, self.F0, self.desired_pos[:3], self.desired_pos[3:6],
                                     self.desired_pos[6:9], self.desired_pos[9:12],
                                     self.model_timestep))
                if self.method == 'euler':
                    ori_error = self.desired_pos[3:6] - ori_real

        if self.method == 'rotation':
            ori_real = T.Rotation_Matrix_To_Vector(self.final_orientation, self.ee_ori_mat)
            ori_error = self.desired_pos[3:6] - ori_real

        vel_ori_error = self.desired_pos[9:12] - self.ee_ori_vel

        # Compute desired force and torque based on errors
        position_error = self.desired_pos[:3].T - self.ee_pos
        vel_pos_error = self.desired_pos[6:9].T - self.ee_pos_vel

        # print(f"self.kp: {self.kp}")
        # print(f"self.kd: {self.kd}")

        #################    calculate PD controller:         #########################################
        desired_force = (np.multiply(np.array(position_error), np.array(self.kp[0:3]))
                         + np.multiply(vel_pos_error, self.kd[0:3]))
        desired_torque = (np.multiply(np.array(ori_error), np.array(self.kp[3:6]))
                          + np.multiply(vel_ori_error, self.kd[3:6]))

        # Compute nullspace matrix (I - Jbar * J) and lambda matrices ((J * M^-1 * J^T)^-1)
        lambda_full, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(self.mass_matrix,
                                                                                 self.J_full,
                                                                                 self.J_pos,
                                                                                 self.J_ori)

        # # Decouples desired positional control from orientation control
        # if self.uncoupling:
        #     decoupled_force = np.dot(lambda_pos, desired_force.T)
        #     decoupled_torque = np.dot(lambda_ori, desired_torque.T)
        #     if self.switch:
        #         decoupled_wrench = np.concatenate([desired_force, desired_torque])
        #     else:
        #         decoupled_wrench = np.concatenate([desired_force, desired_torque])
        #         # decoupled_wrench = np.concatenate([decoupled_force, decoupled_torque])
        # else:
        #     desired_wrench = np.concatenate([desired_force, desired_torque])
        #     decoupled_wrench = np.dot(lambda_full, desired_wrench)

        # Gamma (without null torques) = J^T * F + gravity compensations
        # self.torques = np.zeros(6)
        decoupled_wrench = np.concatenate([desired_force, desired_torque])
        self.torques = np.dot(self.J_full.T, decoupled_wrench).reshape(6, ) + self.torque_compensation

        # Calculate and add nullspace torques (nullspace_matrix^T * Gamma_null) to final torques
        # Note: Gamma_null = desired nullspace pose torques, assumed to be positional joint control relative
        #                     to the initial joint positions
        # self.torques += nullspace_torques(self.mass_matrix, nullspace_matrix,
        #                                   self.initial_joint, self.joint_pos, self.joint_vel)

        self.set_desired_goal = False

        if np.isnan(self.torques).any():
            self.torques = np.zeros(6)

        # for graphs:
        # real_forces = np.dot(np.linalg.inv(self.J_full.T), self.sim.data.qfrc_actuator[:6]).reshape(6, )
        if self.time >= 0:
            self.time_vec.append(self.time)
            self.ee_pos_vec_x.append(self.ee_pos[0])
            self.ee_pos_vec_y.append(self.ee_pos[1])
            self.ee_pos_vec_z.append(self.ee_pos[2])
            self.impedance_model_pos_vec_x.append(self.desired_pos[0])
            self.impedance_model_pos_vec_y.append(self.desired_pos[1])
            self.impedance_model_pos_vec_z.append(self.desired_pos[2])
            self.impedance_ori_vec_x.append(self.desired_pos[3])
            self.impedance_ori_vec_y.append(self.desired_pos[4])
            self.impedance_ori_vec_z.append(self.desired_pos[5])
            self.impedance_vel_vec_x.append(self.desired_pos[6])
            self.impedance_vel_vec_y.append(self.desired_pos[7])
            self.impedance_vel_vec_z.append(self.desired_pos[8])
            self.impedance_ori_vel_vec_x.append(self.desired_pos[9])
            self.impedance_ori_vel_vec_y.append(self.desired_pos[10])
            self.impedance_ori_vel_vec_z.append(self.desired_pos[11])
            self.error_pos_vec.append(ori_error)
            self.ee_ori_vec_x.append(ori_real[0])
            self.ee_ori_vec_y.append(ori_real[1])
            self.ee_ori_vec_z.append(ori_real[2])
            self.desired_torque_x.append(desired_torque[0])
            self.desired_torque_y.append(desired_torque[1])
            self.desired_torque_z.append(desired_torque[2])
            self.desired_force_x.append(desired_force[0])
            self.desired_force_y.append(desired_force[1])
            self.desired_force_z.append(desired_force[2])

            self.ee_sensor_bias_x.append(self.ee_sensor_bias[0])
            self.ee_sensor_bias_y.append(self.ee_sensor_bias[1])
            self.ee_sensor_bias_z.append(self.ee_sensor_bias[2])

            self.Fx_no_bias_sensor_frame.append(self.F_int_no_bias_sensor_frame[0])
            self.Fy_no_bias_sensor_frame.append(self.F_int_no_bias_sensor_frame[1])
            self.Fz_no_bias_sensor_frame.append(self.F_int_no_bias_sensor_frame[2])

            self.Fx_no_bias.append(self.F_int_no_bias[0])
            self.Fy_no_bias.append(self.F_int_no_bias[1])
            self.Fz_no_bias.append(self.F_int_no_bias[2])

            self.wernce_vec_int_Fx.append(self.F_int[0])
            self.wernce_vec_int_Fy.append(self.F_int[1])
            self.wernce_vec_int_Fz.append(self.F_int[2])
            self.wernce_vec_int_Mx.append(self.F_int[3])
            self.wernce_vec_int_My.append(self.F_int[4])
            self.wernce_vec_int_Mz.append(self.F_int[5])
            self.eef_wernce_vec_int_Fx.append(self.sensor_force[0])
            self.eef_wernce_vec_int_Fy.append(self.sensor_force[1])
            self.eef_wernce_vec_int_Fz.append(self.sensor_force[2])
            self.eef_wernce_vec_int_Mx.append(self.sensor_torque[0])
            self.eef_wernce_vec_int_My.append(self.sensor_torque[1])
            self.eef_wernce_vec_int_Mz.append(self.sensor_torque[2])
            # self.real_force_x.append(real_forces[0])
            # self.real_force_y.append(real_forces[1])
            # self.real_force_z.append(real_forces[2])
            self.pos_min_jerk_x.append(self.goal_pos[0])
            self.pos_min_jerk_y.append(self.goal_pos[1])
            self.pos_min_jerk_z.append(self.goal_pos[2])
            if self.method == 'euler':
                self.ori_min_jerk_x.append(ori_min_jerk[0])
                self.ori_min_jerk_y.append(ori_min_jerk[1])
                self.ori_min_jerk_z.append(ori_min_jerk[2])
                self.ori_vel_min_jerk_x.append(ori_vel_min_jerk[0])
                self.ori_vel_min_jerk_y.append(ori_vel_min_jerk[1])
                self.ori_vel_min_jerk_z.append(ori_vel_min_jerk[2])
            else:
                self.ori_min_jerk_x.append(self.goal_ori[0])
                self.ori_min_jerk_y.append(self.goal_ori[1])
                self.ori_min_jerk_z.append(self.goal_ori[2])
                self.ori_vel_min_jerk_x.append(self.goal_ori_vel[0])
                self.ori_vel_min_jerk_y.append(self.goal_ori_vel[1])
                self.ori_vel_min_jerk_z.append(self.goal_ori_vel[2])
            self.vel_min_jerk_x.append(self.goal_vel[0])
            self.vel_min_jerk_y.append(self.goal_vel[1])
            self.vel_min_jerk_z.append(self.goal_vel[2])
            self.ee_vel_vec_x.append((self.ee_pos_vel[0]))
            self.ee_vel_vec_y.append((self.ee_pos_vel[1]))
            self.ee_vel_vec_z.append((self.ee_pos_vel[2]))
            self.ee_ori_vel_vec_x.append(self.ee_ori_vel[0])
            self.ee_ori_vel_vec_y.append(self.ee_ori_vel[1])
            self.ee_ori_vel_vec_z.append(self.ee_ori_vel[2])
        return self.torques

    def has_contacts(self):
        for pair in self.contact_pairs:
            if self.peg_name in pair and self.hole_name in pair:
                return True
        return False

    @property
    def control_limits(self):
        """
        Returns the limits over this controller's action space, overrides the superclass property
        Returns the following (generalized for both high and low limits), based on the impedance mode:

            :Mode `'fixed'`: [joint pos command]
            :Mode `'variable'`: [damping_ratio values, kp values, joint pos command]
            :Mode `'variable_kp'`: [kp values, joint pos command]

        Returns:
            2-tuple:

                - (np.array) minimum action values
                - (np.array) maximum action values
        """
        return self.input_min, self.input_max

    def ImpedanceEq(self, F_int, F0, x0, th0, x0_d, th0_d, dt):
        """
        Impedance Eq: F_int-F0=K(x0-xm)+C(x0_d-xm_d)-Mxm_dd

        Solving the impedance equation for x(k+1)=Ax(k)+Bu(k) where
        x(k+1)=[Xm,thm,Xm_d,thm_d]

        Parameters:
            x0,x0_d,th0,th0_d - desired goal position/orientation and velocity
            F_int - measured force/moments in [N/Nm] (what the robot sense)
            F0 - desired applied force/moments (what the robot does)
            xm_pose - impedance model (updated in a loop) initialized at the initial pose of robot
            A_d, B_d - A and B matrices of x(k+1)=Ax(k)+Bu(k)
        Output:
            X_nex = x(k+1) = [Xm,thm,Xm_d,thm_d]
        """

        # state space formulation
        # X=[xm;thm;xm_d;thm_d] U=[F_int;M_int;x0;th0;x0d;th0d]
        A_1 = np.concatenate((np.zeros([6, 6], dtype=int), np.identity(6)), axis=1)
        A_2 = np.concatenate((np.dot(-np.linalg.pinv(self.M), self.K), np.dot(-np.linalg.pinv(self.M), self.C)), axis=1)
        A_temp = np.concatenate((A_1, A_2), axis=0)

        B_1 = np.zeros([6, 18], dtype=int)
        B_2 = np.concatenate((np.linalg.pinv(self.M), np.dot(np.linalg.pinv(self.M), self.K),
                              np.dot(np.linalg.pinv(self.M), self.C)), axis=1)
        B_temp = np.concatenate((B_1, B_2), axis=0)

        if np.isnan(A_temp).any() or np.isnan(B_temp).any():
            s = 1
        # discrete state space A, B matrices
        A_d = expm(A_temp * dt)
        B_d = np.dot(np.dot(np.linalg.pinv(A_temp), (A_d - np.identity(A_d.shape[0]))), B_temp)

        # impedance model xm is initialized to initial position of the EEF and modified by force feedback
        xm = self.impedance_vec[:3].reshape(3, 1)
        thm = self.impedance_vec[3:6].reshape(3, 1)
        xm_d = self.impedance_vec[6:9].reshape(3, 1)
        thm_d = self.impedance_vec[9:12].reshape(3, 1)

        # State Space vectors
        X = np.concatenate((xm, thm, xm_d, thm_d), axis=0)  # 12x1 column vector

        U = np.concatenate((-F0 + F_int, x0, th0, x0_d, th0_d), axis=0).reshape(18, 1)

        # discrete state solution X(k+1)=Ad*X(k)+Bd*U(k)
        X_nex = np.dot(A_d, X) + np.dot(B_d, U)
        # print(X_nex[9:12])
        self.impedance_vec = deepcopy(X_nex)
        return X_nex.reshape(12, )

    def set_control_param(self, action):
        # self.learn += status
        #   TODO - set_control_param(self, action) - add cases of different learning- just K; K+C ect.
        # if self.learn == 1:
        if self.control_dim == 36:
            self.K = action.reshape(6, 6)

            self.C = np.array([[0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0]])

        if self.control_dim == 6:
            self.K = np.array([[action[0], action[1], action[2], action[3], action[4], action[5]],
                               [action[1], action[2], action[3], action[4], action[5], action[0]],
                               [action[2], action[3], action[4], action[5], action[0], action[1]],
                               [action[3], action[4], action[5], action[0], action[1], action[2]],
                               [action[4], action[5], action[0], action[1], action[2], action[3]],
                               [action[5], action[0], action[1], action[2], action[3], action[4]]])

        if self.control_dim == 19:
            self.K = np.array([[action[0], action[14], action[1], action[16], action[2], action[17]],
                               [action[14], action[3], action[4], action[5], action[17], action[18]],
                               [action[15], action[16], action[6], action[17], action[18], action[14]],
                               [action[16], action[7], action[8], action[9], action[14], action[15]],
                               [action[10], action[17], action[11], action[14], action[12], action[16]],
                               [action[17], action[18], action[14], action[15], action[16], action[13]]])

            self.C = np.array([[0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0]])
            # self.ee_sensor_bias = deepcopy(self.sim.data.sensordata)
        if self.control_dim == 72:
            k = action[:len(action) // 2]
            self.K = k.reshape(6, 6)
            c = action[len(action) // 2:]
            self.C = c.reshape(6, 6)

        if self.control_dim == 36:
            self.K = np.array([[action[0], 0, 0, 0, action[1], 0],
                               [0, action[2], 0, action[3], 0, 0],
                               [0, 0, action[4], 0, 0, 0],
                               [0, action[5], 0, action[6], 0, 0],
                               [action[7], 0, 0, 0, action[8], 0],
                               [0, 0, 0, 0, 0, action[9]]])

            self.C = np.array([[action[10], 0, 0, 0, action[11], 0],
                               [0, action[12], 0, action[13], 0, 0],
                               [0, 0, action[14], 0, 0, 0],
                               [0, action[15], 0, action[16], 0, 0],
                               [action[17], 0, 0, 0, action[18], 0],
                               [0, 0, 0, 0, 0, action[19]]])

            self.M = np.array([[action[20], 0, 0, 0, action[21], 0],
                               [0, action[22], 0, action[23], 0, 0],
                               [0, 0, action[24], 0, 0, 0],
                               [0, action[25], 0, action[26], 0, 0],
                               [action[27], 0, 0, 0, action[28], 0],
                               [0, 0, 0, 0, 0, action[29]]])
            self.kp_impedance = action[-6:]

        if self.control_dim == 24:
            self.K = np.array([[action[0], 0, 0, 0, 0, 0],
                               [0, action[1], 0, 0, 0, 0],
                               [0, 0, action[2], 0, 0, 0],
                               [0, 0, 0, action[3], 0, 0],
                               [0, 0, 0, 0, action[4], 0],
                               [0, 0, 0, 0, 0, action[5]]])

            self.C = np.array([[action[6], 0, 0, 0, 0, 0],
                               [0, action[7], 0, 0, 0, 0],
                               [0, 0, action[8], 0, 0, 0],
                               [0, 0, 0, action[9], 0, 0],
                               [0, 0, 0, 0, action[10], 0],
                               [0, 0, 0, 0, 0, action[11]]])

            self.M = np.array([[action[12], 0, 0, 0, 0, 0],
                               [0, action[13], 0, 0, 0, 0],
                               [0, 0, action[14], 0, 0, 0],
                               [0, 0, 0, action[15], 0, 0],
                               [0, 0, 0, 0, action[16], 0],
                               [0, 0, 0, 0, 0, action[17]]])

            self.kp_impedance = action[-6:]
            self.kd_impedance = 2 * np.sqrt(self.kp_impedance) * np.sqrt(2)

        if self.control_dim == 31:
            self.K = np.array([[action[0], 0, action[1], 0, action[2], 0],
                               [0, action[3], action[4], action[5], 0, 0],
                               [0, 0, action[6], 0, 0, 0],
                               [0, action[7], 0, action[8], 0, 0],
                               [action[9], 0, 0, 0, action[10], 0],
                               [0, 0, 0, 0, 0, action[11]]])

            self.C = np.array([[action[12], 0, action[13], 0, action[14], 0],
                               [0, action[15], action[16], action[17], 0, 0],
                               [0, 0, action[18], 0, 0, 0],
                               [0, action[19], 0, action[20], 0, 0],
                               [action[21], 0, 0, 0, action[22], 0],
                               [0, 0, 0, 0, 0, action[23]]])

            self.M = deepcopy(self.mass_matrix)

            self.kp_impedance = action[24:30]
            self.kd_impedance = action[30:]

        if self.control_dim == 40:
            self.K = action[:36].reshape(6, 6)
            self.C = np.array([[0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0]])
            self.kp_impedance[:3] = self.nums2array(action[-4], 3)
            self.kp_impedance[3:] = self.nums2array(action[-3], 3)
            self.kd_impedance[:3] = self.nums2array(action[-2], 3)
            self.kd_impedance[3:] = self.nums2array(action[-1], 3)

        if self.control_dim == 18:
            self.K = np.array([[abs(action[0]), 0, 0, 0, 0, 0],
                               [0, abs(action[1]), 0, 0, 0, 0],
                               [0, 0, abs(action[2]), 0, 0, 0],
                               [0, 0, 0, abs(action[3]), 0, 0],
                               [0, 0, 0, 0, abs(action[4]), 0],
                               [0, 0, 0, 0, 0, abs(action[5])]])

            self.C = np.array([[abs(action[6]), 0, 0, 0, 0, 0],
                               [0, abs(action[7]), 0, 0, 0, 0],
                               [0, 0, abs(action[8]), 0, 0, 0],
                               [0, 0, 0, abs(action[9]), 0, 0],
                               [0, 0, 0, 0, abs(action[10]), 0],
                               [0, 0, 0, 0, 0, abs(action[11])]])

            self.M = np.array([[abs(action[12]), 0, 0, 0, 0, 0],
                               [0, abs(action[13]), 0, 0, 0, 0],
                               [0, 0, abs(action[14]), 0, 0, 0],
                               [0, 0, 0, abs(action[15]), 0, 0],
                               [0, 0, 0, 0, abs(action[16]), 0],
                               [0, 0, 0, 0, 0, abs(action[17])]])

            self.kp_impedance = np.array([700., 500., 100., 450., 450., 450.])
            self.kd_impedance = 2 * np.sqrt(self.kp_impedance) * np.sqrt(2)
            self.kd_impedance[3:] = 30

        if self.control_dim == 30:
            self.K = np.array([[abs(action[0]), 0, 0, 0, action[1], 0],
                               [0, abs(action[2]), 0, action[3], 0, 0],
                               [0, 0, abs(action[4]), 0, 0, 0],
                               [0, action[5], 0, abs(action[6]), 0, 0],
                               [action[7], 0, 0, 0, abs(action[8]), 0],
                               [0, 0, 0, 0, 0, abs(action[9])]])

            self.C = np.array([[abs(action[10]), 0, 0, 0, action[11], 0],
                               [0, abs(action[12]), 0, action[13], 0, 0],
                               [0, 0, abs(action[14]), 0, 0, 0],
                               [0, action[15], 0, abs(action[16]), 0, 0],
                               [action[17], 0, 0, 0, abs(action[18]), 0],
                               [0, 0, 0, 0, 0, abs(action[19])]])

            self.M = np.array([[abs(action[20]), 0, 0, 0, action[21], 0],
                               [0, abs(action[22]), 0, action[23], 0, 0],
                               [0, 0, abs(action[24]), 0, 0, 0],
                               [0, action[25], 0, abs(action[26]), 0, 0],
                               [action[27], 0, 0, 0, abs(action[28]), 0],
                               [0, 0, 0, 0, 0, abs(action[29])]])
            # self.C = np.nan_to_num(2 * np.sqrt(np.dot(self.K, self.M)))
            self.kp_impedance = np.array([700., 500., 100., 450., 450., 450.])
            self.kd_impedance = 2 * np.sqrt(self.kp_impedance) * np.sqrt(2)
            self.kd_impedance[3:] = 30

        if self.control_dim == 26:
            # self.K = np.genfromtxt(r"C:\Users\z2mdxf\Documents\Ori\Maagad\RL_gym\Matrices\K_mean.csv", delimiter=',')
            # self.C = np.genfromtxt(r"C:\Users\z2mdxf\Documents\Ori\Maagad\RL_gym\Matrices\C_mean.csv", delimiter=',')
            # self.M = np.genfromtxt(r"C:\Users\z2mdxf\Documents\Ori\Maagad\RL_gym\Matrices\M_mean.csv", delimiter=',')
            self.K = np.array([[abs(action[0]), 0, 0, 0, action[1], 0],
                               [0, abs(action[2]), 0, action[3], 0, 0],
                               [0, 0, abs(action[4]), 0, 0, 0],
                               [0, action[5], 0, abs(action[6]), 0, 0],
                               [action[7], 0, 0, 0, abs(action[8]), 0],
                               [0, 0, 0, 0, 0, abs(action[9])]])

            self.C = np.array([[abs(action[10]), 0, 0, 0, action[11], 0],
                               [0, abs(action[12]), 0, action[13], 0, 0],
                               [0, 0, abs(action[14]), 0, 0, 0],
                               [0, action[15], 0, abs(action[16]), 0, 0],
                               [action[17], 0, 0, 0, abs(action[18]), 0],
                               [0, 0, 0, 0, 0, abs(action[19])]])

            self.M = np.array([[abs(action[20]), 0, 0, 0, 0, 0],
                               [0, abs(action[21]), 0, 0, 0, 0],
                               [0, 0, abs(action[22]), 0, 0, 0],
                               [0, 0, 0, abs(action[23]), 0, 0],
                               [0, 0, 0, 0, abs(action[24]), 0],
                               [0, 0, 0, 0, 0, abs(action[25])]])

            self.K = np.array([[24.51158142, 0., 0., 0., -42.63611603, 0.],
                               [0., 40.25193405, 0., 26.59643364, 0., 0.],
                               [0., 0., 23.69382477, 0., 0., 0.],
                               [0., -10.66889191, 0., 3.27396274, 0., 0.],
                               [-19.22241402, 0., 0., 0., 46.74688339, 0.],
                               [0., 0., 0., 0., 0., 24.80905724]])

            self.C = np.array([[66.41168976, 0., 0., 0., -26.20734787, 0.],
                               [0., 98.06533051, 0., 26.14341736, 0., 0.],
                               [0., 0., 103.66620636, 0., 0., 0.],
                               [0., 21.57993698, 0., 55.37984467, 0., 0.],
                               [-0.29122657, 0., 0., 0., 0.14158408, 0.],
                               [0., 0., 0., 0., 0., 2.37160134]])

            self.M = np.array([[112.29223633, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 72.80897522, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 169.45898438, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 37.9505806, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 4.87572193, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 14.63672161]])

            # self.C = np.array([[66.64912796, 0., 0., 0., -26.30130032, 0.],
            #                    [0., 98.41663391, 0., 26.23718006, 0., 0.],
            #                    [0., 0., 104.03779221, 0., 0., 0.],
            #                    [0., 21.65733665, 0., 55.57846291, 0., 0.],
            #                    [-0.29211203, 0., 0., 0., 0.14163978, 0.],
            #                    [0., 0., 0., 0., 0., 2.37964597]])
            # self.K = np.array([[24.59960964, 0., 0., 0., -42.78901123, 0.],
            #                    [0., 40.39589363, 0., 26.69166386, 0., 0.],
            #                    [0., 0., 23.77889698*2, 0., 0., 0.],
            #                    [0., -10.7071373, 0., 3.28527857, 0., 0.],
            #                    [-19.29147655, 0., 0., 0., 46.91463356, 0.],
            #                    [0., 0., 0., 0., 0., 24.89790821]])
            # self.M = np.array([[112.69425827, 0., 0., 0., 0., 0.],
            #                    [0., 73.06942383, 0., 0., 0., 0.],
            #                    [0., 0., 170.06573547 * 0.05, 0., 0., 0.],
            #                    [0., 0., 0., 38.08610428, 0., 0.],
            #                    [0., 0., 0., 0., 4.89339188, 0.],
            #                    [0., 0., 0., 0., 0., 14.68913105]])

            # self.K = np.array([[0.54285342, 0., 0., 0., -2.28454328, 0.],
            #                    [0., 0.980474, 0., 2.69239235, 0., 0.],
            #                    [0., 0., 0.82381952, 0., 0., 0.],
            #                    [0., -0.35678336, 0., 0.28500804, 0., 0.],
            #                    [-0.69362265, 0., 0., 0., 0.42916226, 0.],
            #                    [0., 0., 0., 0., 0., 0.71959662]])
            #
            # self.C = np.array([[0.51152909, 0., 0., 0., 0.99223411, 0.],
            #                    [0., 0.47802943, 0., 0.16433489, 0., 0.],
            #                    [0., 0., 1.07876301, 0., 0., 0.],
            #                    [0., 1.03930426, 0., 0.67218661, 0., 0.],
            #                    [0.33381957, 0., 0., 0., 0.21643667, 0.],
            #                    [0., 0., 0., 0., 0., 0.22787397]])
            #
            # self.M = np.array([[0.18114398, 0., 0., 0., 0., 0.],
            #                    [0., 0.64173299, 0., 0., 0., 0.],
            #                    [0., 0., 1.93259776, 0., 0., 0.],
            #                    [0., 0., 0., 0.46770465, 0., 0.],
            #                    [0., 0., 0., 0., 0.52367067, 0.],
            #                    [0., 0., 0., 0., 0., 0.45890355]])

            # print(f"self.C: {self.C}")
            # print(f"self.K: {self.K}")
            # print(f"self.M: {self.M}")

            self.kp_impedance = np.array([700., 500., 100., 450., 450., 450.])
            self.kd_impedance = 2 * np.sqrt(self.kp_impedance) * np.sqrt(2)
            self.kd_impedance[3:] = 30

        if self.control_dim == 108:
            # self.K = np.load('environments/Kimp.npy')
            # self.K = np.array([[100, 0., 0., 0., 0., 0.],
            #                    [0., 117.6, 0., 0., 0., 0.],
            #                    [0., 0., 800., 0., 0., 0.],
            #                    [0., 0., 0., 300., 0., 0.],
            #                    [0., 0., 0., 0., 235, 0.],
            #                    [0., 0., 0., 0., 0., 20.]])

            self.K = np.array([[49.20809937, 0., 0., 0., 42.583914, 0.],
                               [0., 1.15867949, 0., 24.9493, 0., 0.],
                               [0., 0., 32.7383461, 0., 0., 0.],
                               [0., -16.162, 0., 82.36485291, 0., 0.],
                               [-50.216, 0., 0., 0., 119.234375, 0.],
                               [0., 0., 0., 0., 0., 23.38664055]])

            # self.C = 2 * np.sqrt(self.K)*np.sqrt(2)
            # self.C = np.array([[50, 0., 0., 0., 0., 0.],
            #                    [0., 90, 0., 0., 0., 0.],
            #                    [0., 0., 20., 0., 0., 0.],
            #                    [0., 0., 0., 50, 0., 0.],
            #                    [0., 0., 0., 0., 45, 0.],
            #                    [0., 0., 0., 0., 0., 7.]])
            self.C = np.array([[118.36997, 0., 0., 0., -86.329, 0.],
                               [0., 17.7242, 0., -42.90356, 0., 0.],
                               [0., 0., 47.8301, 0., 0., 0.],
                               [0., 25.0288, 0., 53.0588, 0., 0.],
                               [-14.4582, 0., 0., 0., 28.0303, 0.],
                               [0., 0., 0., 0., 0., 71.2552]])
            # self.C[4, 0] = 1
            # self.C[3, 3] = 100
            # self.C[0, 0] = 6
            # self.C[3,1] = 10
            # self.M = np.load('environments/Mimp.npy')
            self.M = np.array([[52.91357, 0., 0., 0., 0., 0.],
                               [0., 24.43817, 0., 0., 0., 0.],
                               [0., 0., 9.51036, 0., 0., 0.],
                               [0., 0., 0., 128.93762, 0., 0.],
                               [0., 0., 0., 0., 29.74888, 0.],
                               [0., 0., 0., 0., 0., 56.30805]])
            self.kp_impedance = np.array([700., 500., 100., 450., 450., 450.])
            self.kd_impedance = 2 * np.sqrt(self.kp_impedance) * np.sqrt(2)
            self.kd_impedance[3:] = 30

    def save_plot_data(self):
        data = {}
        data["time"] = self.time_vec

        data["Xm position"] = self.impedance_model_pos_vec_x
        data["Xr position"] = self.ee_pos_vec_x
        data["X_ref position"] = self.pos_min_jerk_x
        data["Ym position"] = self.impedance_model_pos_vec_y
        data["Yr position"] = self.ee_pos_vec_y
        data["Y_ref position"] = self.pos_min_jerk_y
        data["Zm position"] = self.impedance_model_pos_vec_z
        data["Zr position"] = self.ee_pos_vec_z
        data["Z_ref position"] = self.pos_min_jerk_z

        data["Fx"] = self.wernce_vec_int_Fx
        data["Fx_des"] = self.desired_force_x
        data["Fy"] = self.wernce_vec_int_Fy
        data["Fy_des"] = self.desired_force_y
        data["Fz"] = self.wernce_vec_int_Fz
        data["Fz_des"] = self.desired_force_z

        data["Mx"] = self.wernce_vec_int_Mx
        data["mx_des"] = self.desired_torque_x
        data["My"] = self.wernce_vec_int_My
        data["my_des"] = self.desired_torque_y
        data["Mz"] = self.wernce_vec_int_Mz
        data["mz_des"] = self.desired_torque_z

        data["Xm vel [m/s]"] = self.impedance_vel_vec_x
        data["Xr vel [m/s]"] = self.ee_vel_vec_x
        data["X_ref vel [m/s]"] = self.vel_min_jerk_x
        data["Ym vel [m/s]"] = self.impedance_vel_vec_y
        data["Yr vel [m/s]"] = self.ee_vel_vec_y
        data["Y_ref vel [m/s]"] = self.vel_min_jerk_y
        data["Zm vel [m/s]"] = self.impedance_vel_vec_z
        data["Zr vel [m/s]"] = self.ee_vel_vec_z
        data["Z_ref vel [m/s]"] = self.vel_min_jerk_z

        data["Xr ori vel [rad/s]"] = self.ee_ori_vel_vec_x
        data["X_ref ori vel [rad/s]"] = self.ori_vel_min_jerk_x
        data["Yr ori vel [rad/s]"] = self.ee_ori_vel_vec_y
        data["Y_ref ori vel [rad/s]"] = self.ori_vel_min_jerk_y
        data["Zr ori vel [rad/s]"] = self.ee_ori_vel_vec_z
        data["Z_ref ori vel [rad/s]"] = self.ori_vel_min_jerk_z

        df = pd.DataFrame(data)
        df.to_csv("data.csv", index=False)

    def control_plotter(self):
        t = self.time_vec  # list(range(0, np.size(self.ee_pos_vec_x)))
        self.save_plot_data()
        ################################################################################################################
        plt.figure()
        ax1 = plt.subplot(311)
        ax1.plot(t, self.impedance_model_pos_vec_x, 'g--', label='Xm position')
        ax1.plot(t, self.ee_pos_vec_x, 'b', label='Xr position')
        ax1.plot(t, self.pos_min_jerk_x, 'r--', label='X_ref position')
        ax1.legend()
        ax1.set_title('X Position [m]')

        ax2 = plt.subplot(312)
        ax2.plot(t, self.impedance_model_pos_vec_y, 'g--', label='Ym position')
        ax2.plot(t, self.ee_pos_vec_y, 'b', label='Yr position')
        ax2.plot(t, self.pos_min_jerk_y, 'r--', label='Y_ref position')
        ax2.legend()
        ax2.set_title('Y Position [m]')

        ax3 = plt.subplot(313)
        ax3.plot(t, self.impedance_model_pos_vec_z, 'g--', label='Zm position')
        ax3.plot(t, self.ee_pos_vec_z, 'b', label='Zr position')
        ax3.plot(t, self.pos_min_jerk_z, 'r--', label='Z_ref position')
        ax3.legend()
        ax3.set_title('Z Position [m]')

        ################################################################################################################
        plt.figure()
        ax1 = plt.subplot(311)
        ax1.plot(t, self.impedance_vel_vec_x, 'g--', label='Xm vel')
        ax1.plot(t, self.ee_vel_vec_x, 'b', label='Xr vel')
        ax1.plot(t, self.vel_min_jerk_x, 'r--', label='X_ref vel')
        ax1.legend()
        ax1.set_title('X Velocity [m/s]')

        ax2 = plt.subplot(312)
        ax2.plot(t, self.impedance_vel_vec_y, 'g--', label='Ym vel')
        ax2.plot(t, self.ee_vel_vec_y, 'b', label='Yr vel')
        ax2.plot(t, self.vel_min_jerk_y, 'r--', label='Y_ref vel')
        ax2.legend()
        ax2.set_title('Y Velocity [m/s]')

        ax3 = plt.subplot(313)
        ax3.plot(t, self.impedance_vel_vec_z, 'g--', label='Zm vel')
        ax3.plot(t, self.ee_vel_vec_z, 'b', label='Zr vel')
        ax3.plot(t, self.vel_min_jerk_z, 'r--', label='Z_ref vel')
        ax3.legend()
        ax3.set_title('Z Velocity [m/s]')

        ################################################################################################################
        plt.figure()
        ax1 = plt.subplot(311)
        ax1.plot(t, self.ee_ori_vel_vec_x, 'b', label='Xr')
        ax1.plot(t, self.ori_vel_min_jerk_x, 'r--', label='X_ref ')
        ax1.legend()
        ax1.set_title('X ori vel [rad/s]')

        ax2 = plt.subplot(312)
        ax2.plot(t, self.ee_ori_vel_vec_y, 'b', label='Yr ')
        ax2.plot(t, self.ori_vel_min_jerk_y, 'r--', label='Y_ref ')
        ax2.legend()
        ax2.set_title('Y ori vel [rad/s]')

        ax3 = plt.subplot(313)
        ax3.plot(t, self.ee_ori_vel_vec_z, 'b', label='Zr ')
        ax3.plot(t, self.ori_vel_min_jerk_z, 'r--', label='Z_ref ')
        ax3.legend()
        ax3.set_title('Z ori vel [rad/s]')

        ################################################################################################################
        plt.figure()
        ax1 = plt.subplot(311)
        ax1.plot(t, self.ee_ori_vec_x, 'b', label='Xr')
        ax1.plot(t, self.ori_min_jerk_x, 'r', label='X_ref ')
        ax1.plot(t, self.impedance_ori_vec_x, 'g--', label='Xm ')
        ax1.legend()
        ax1.set_title('X ori [rad]')

        ax2 = plt.subplot(312)
        ax2.plot(t, self.ee_ori_vel_vec_y, 'b', label='Yr ')
        ax2.plot(t, self.ori_min_jerk_y, 'r', label='Y_ref ')
        ax2.plot(t, self.impedance_ori_vec_y, 'g--', label='Ym ')
        ax2.legend()
        ax2.set_title('Y ori [rad]')

        ax3 = plt.subplot(313)
        ax3.plot(t, self.ee_ori_vec_z, 'b', label='Zr ')
        ax3.plot(t, self.ori_min_jerk_z, 'r', label='Z_ref ')
        ax3.plot(t, self.impedance_ori_vec_z, 'g--', label='Zm ')
        ax3.legend()
        ax3.set_title('Z ori [rad]')

        ################################################################################################################
        plt.figure()
        ax1 = plt.subplot(311)
        ax1.plot(t, self.wernce_vec_int_Fx, 'b', label='Fx')
        # ax1.plot(t, self.Fx_no_bias, 'r', label='Fx no bias')
        # ax1.plot(t, self.Fx_no_bias_sensor_frame, 'y', label='Fx no bias (sensor)')
        # ax1.plot(t, self.ee_sensor_bias_x, 'c', label='Bias')
        ax1.plot(t, self.desired_force_x, 'g', label='Fx_des')
        ax1.legend()
        ax1.set_title('Fx [N]')

        ax2 = plt.subplot(312)
        ax2.plot(t, self.wernce_vec_int_Fy, 'b', label='Fy')
        # ax2.plot(t, self.Fy_no_bias, 'r', label='Fy no bias')
        # ax2.plot(t, self.ee_sensor_bias_y, 'c', label='Bias')
        # ax2.plot(t, self.Fy_no_bias_sensor_frame, 'y', label='Fy no bias (sensor)')
        ax2.plot(t, self.desired_force_y, 'g', label='Fy_des')
        ax2.legend()
        ax2.set_title('Fy [N]')

        ax3 = plt.subplot(313)
        # ax3.plot(t, self.ee_sensor_bias_z, 'c', label='Bias')
        ax3.plot(t, self.wernce_vec_int_Fz, 'b', label='Fz')
        # ax3.plot(t, self.Fz_no_bias, 'r', label='Fz no bias')
        # ax3.plot(t, self.Fz_no_bias_sensor_frame, 'y', label='Fz no bias (sensor)')
        ax3.plot(t, self.desired_force_z, 'g', label='Fz_des')
        ax3.legend()
        ax3.set_title('Fz [N]')

        ################################################################################################################
        plt.figure()
        ax1 = plt.subplot(311)
        ax1.plot(t, self.wernce_vec_int_Mx, 'b', label='Mx')
        ax1.plot(t, self.desired_torque_x, 'g', label='mx_des')
        ax1.legend()
        ax1.set_title('Mx [Nm]')

        ax2 = plt.subplot(312)
        ax2.plot(t, self.wernce_vec_int_My, 'b', label='My')
        ax2.plot(t, self.desired_torque_y, 'g', label='My_des')
        ax2.legend()
        ax2.set_title('My [Nm]')

        ax3 = plt.subplot(313)
        ax3.plot(t, self.wernce_vec_int_Mz, 'b', label='Mz')
        ax3.plot(t, self.desired_torque_z, 'g', label='Mz_des')
        ax3.legend()
        ax3.set_title('Mz [Nm]')
        plt.show()