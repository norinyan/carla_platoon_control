# -*- coding: utf-8 -*-

import numpy as np
import cvxpy as cp
from math import sin, cos, atan2, radians
from .base_controller import LateralController
from .base_controller import LeaderLonController
from .base_controller import FollowerLonController

# 横向控制
class LatMPC(LateralController):
    def __init__(self, params):
        self.N  = params.get("N",  15)     # 预测步长
        self.dt = params.get("dt",  0.1)

        self.L            = params.get("wheelbase",    2.85)
        self.max_steer_rad = radians(params.get("max_steer_deg", 35.0))

        # 代价权重
        q_ey          = params.get("q_ey",    1.0)
        q_eyaw        = params.get("q_eyaw",  2.0)
        self.Q        = np.diag([q_ey, q_eyaw])
        self.P        = np.diag([q_ey * 2.0, q_eyaw * 2.0])  # 终端代价适当加大
        self.R_delta  = params.get("R_delta",  0.1)
        self.R_ddelta = params.get("R_ddelta", 0.5)

        # 转角约束
        self.delta_min  = -self.max_steer_rad
        self.delta_max  =  self.max_steer_rad
        self.ddelta_min = -radians(params.get("ddelta_max_deg", 5.0))
        self.ddelta_max =  radians(params.get("ddelta_max_deg", 5.0))

        self.delta_prev = 0.0

        self._setup_mpc()


    def compute(self, vehicle_state, ref_points, dt):
        """
        vehicle_state : {x, y, yaw, speed}
        ref_points    : 从最近点开始往前的 N 个参考点，每点含 {x, y, yaw, kappa, speed, s}
        dt            : 控制周期 (s)
        return        : steer，范围 [-1.0, 1.0]
        """
        self.dt             = dt
        self.vehicle_state  = vehicle_state
        self.ref_points     = ref_points

        # 1. 计算误差状态
        e_y, e_yaw = self._compute_error()

        # 2. 求解MPC输出
        delta_opt = self._solve_mpc(e_y, e_yaw)

        # 3. 更新历史
        self.delta_prev = delta_opt

        # 4. 输出steer
        steer = delta_opt / self.max_steer_rad
        steer = max(-1.0, min(1.0, steer))
        return steer
        
    def _compute_error(self):
        """
        计算当前横向偏差 e_y 和航向偏差 e_yaw。
        使用最近参考点（ref_points[0]）作为误差计算基准。

        e_y   : 车辆位置投影到参考点法向的距离（左正右负）
        e_yaw : 车辆航向 - 参考航向，归一化到 [-pi, pi]
        """
        x,   y   = self.vehicle_state["x"],   self.vehicle_state["y"]
        yaw       = self.vehicle_state["yaw"]
        ref        = self.ref_points[0]
        x_ref,  y_ref   = ref["x"], ref["y"]
        yaw_ref       = ref["yaw"]

        dx  = x - x_ref
        dy  = y - y_ref
        e_y = -dx * sin(yaw_ref) + dy * cos(yaw_ref)

        e_yaw = yaw - yaw_ref
        e_yaw = atan2(sin(e_yaw), cos(e_yaw))

        return e_y, e_yaw

    def _solve_mpc(self, e_y, e_yaw):
       
        # 更新初始状态和模型矩阵
        A, B = self._build_model()
        self.x0_param.value         = np.array([e_y, e_yaw])
        self.A_param.value          = A
        self.B_param.value          = B
        self.delta_prev_param.value = self.delta_prev 

        # 求解
        try:
            self.mpc_problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except cp.error.SolverError:
            print("[MPC] 求解异常，保持上一帧")
            return self.delta_prev
        
        # 保底
        if self.u_var.value is None:
            print("[MPC] 求解失败，保持上一帧")
            return self.delta_prev

        delta_opt = float(self.u_var.value[0, 0])

        # 取第一步最优前轮转角
        return delta_opt

    def _build_model(self):
        v = max(self.vehicle_state["speed"], 0.5)
        A = np.array([[1.0, v * self.dt],
                    [0.0, 1.0       ]])
        B = np.array([[0.0                  ],
                    [- v * self.dt / self.L ]])
        return A, B

    def _setup_mpc(self):

        # 1: 声明优化变量   
        self.x_var = cp.Variable((2, self.N + 1))   # 状态序列 [e_y, e_yaw]
        self.u_var = cp.Variable((1, self.N))        # 控制序列 [delta]
        
        # 2: 声明每帧更新的参数
        self.x0_param         = cp.Parameter(2)        # 初始误差状态
        self.A_param          = cp.Parameter((2, 2))   # 状态矩阵
        self.B_param          = cp.Parameter((2, 1))   # 控制矩阵
        self.delta_prev_param = cp.Parameter()         # 上一帧转角

        # 3: 构建代价函数和约束

        cost        = 0
        constraints = [self.x_var[:, 0] == self.x0_param]  # 初始状态约束

        for k in range(self.N):
            # 转角增量（k=0时参考上一帧，k>0时参考上一步）
            delta_ref = self.delta_prev_param if k == 0 else self.u_var[0, k - 1]
            ddelta    = self.u_var[0, k] - delta_ref

            # 代价：状态偏差 + 转角幅值 + 转角增量
            cost += cp.quad_form(self.x_var[:, k], self.Q)
            cost += self.R_delta  * cp.square(self.u_var[0, k])
            cost += self.R_ddelta * cp.square(ddelta)

            # 约束：动力学 + 转角幅值 + 转角增量
            constraints += [self.x_var[:, k + 1] == self.A_param @ self.x_var[:, k] + self.B_param @ self.u_var[:, k]]
            constraints += [self.u_var[0, k] >= self.delta_min, self.u_var[0, k] <= self.delta_max]
            constraints += [ddelta >= self.ddelta_min,          ddelta <= self.ddelta_max]

        # 终端代价
        cost += cp.quad_form(self.x_var[:, self.N], self.P)

        # 4: 构建问题
        self.mpc_problem = cp.Problem(cp.Minimize(cost), constraints)


# 纵向Leader
class LeaderLonPID(LeaderLonController):
    def __init__(self, params):

        self.kp = params.get("kp", 0.55)
        self.ki = params.get("ki", 0.01)
        self.kd = params.get("kd", 0.05)

        self.a_limit = params.get("a_limit", 3.0)  # 最大加速度
        self.i_limit = params.get("i_limit", 1.0)  # 积分限幅
        
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, vehicle_state, ref_points, dt):
        
        v = vehicle_state["speed"]
        ref_point = ref_points[0]
        v_ref = ref_point["speed"]
        
        dt = max(dt, 1e-3)
        
        # 速度误差
        speed_err = v_ref - v

        # 积分项（限幅防止饱和）
        self.integral += speed_err * dt
        self.integral = max(-self.i_limit, min(self.i_limit, self.integral))
        
        # 微分项
        d_error = (speed_err - self.prev_error) / dt
        self.prev_error = speed_err
        
        # PID
        a_cmd = self.kp * speed_err + self.ki * self.integral + self.kd * d_error
        
        # 限幅
        a_cmd = max(-self.a_limit, min(self.a_limit, a_cmd))
        
        return a_cmd

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0


# 纵向Follower
class FollowerLonMPC(FollowerLonController):
    def __init__(self, params):
        self.N  = params.get("N",  10)     # 预测步长
        self.dt = params.get("dt",  0.1)

        # DTH 动态时距参数
        self.h0     = params.get("h0",     1.0)
        self.gamma  = params.get("gamma",  3.3)
        self.h_min  = params.get("h_min",  0.6)
        self.h_max  = params.get("h_max",  1.5)
        self.d0     = params.get("d0",     5.0)

        # 代价权重
        q_s       = params.get("q_s",  4.0)
        q_v       = params.get("q_v",  1.0)
        self.Q    = np.diag([q_s, q_v])
        self.R_u  = params.get("r_u",  0.2)
        self.R_du = params.get("r_du", 0.5)

        # 物理约束
        self.a_min   = params.get("a_min",   -2.0)
        self.a_max   = params.get("a_max",    1.5)
        self.v_min   = params.get("v_min",    0.0)
        self.v_max   = params.get("v_max",    8.0)
        self.gap_min = params.get("gap_min",  5.0)
        self.gap_max = params.get("gap_max", 30.0)

        # Tube 半径参数
        self.omega_s0    = params.get("omega_s0",    0.5)
        self.omega_v0    = params.get("omega_v0",    0.2)
        self.beta_s      = params.get("beta_s",      3.0)
        self.beta_v      = params.get("beta_v",      1.0)
        self.omega_s_min = params.get("omega_s_min", 0.3)
        self.omega_s_max = params.get("omega_s_max", 3.0)
        self.omega_v_min = params.get("omega_v_min", 0.1)
        self.omega_v_max = params.get("omega_v_max", 1.5)

        # Tube 反馈增益
        k_s    = params.get("k_s", 0.2)
        k_v    = params.get("k_v", 0.6)
        self.K = np.array([k_s, k_v])

        # 事件触发参数，第三步先读取但不启用
        self.sigma      = params.get("sigma",      0.8)
        self.tau_thresh = params.get("tau_thresh", 0.07)

        self.u_prev = 0.0
        self.u_bar_seq_last = None
        self.x_bar_seq_last = None
        self.trigger_hold_index = 1

        self._setup_mpc()

    def compute(self, vehicle_state, ref_points, front_state, tau, dt):
        """
        vehicle_state : 当前跟随车状态，包含 x, y, yaw, speed, s
        ref_points    : 从最近点开始往前的参考点，第三步暂不直接使用
        front_state   : 前车状态，包含 x, y, yaw, speed, s
        tau           : 前车到当前跟随车的单向通信延迟，单位秒
        dt            : 控制周期 (s)
        return        : a_cmd, debug_info
        """
        self.dt            = dt
        self.vehicle_state = vehicle_state
        self.ref_points    = ref_points
        self.front_state   = front_state
        self.tau           = tau

        # 1. 计算误差状态
        x, h, desired_gap, gap, e_s, e_v = self._compute_error()

        # 2. 计算Tube半径
        omega_s, omega_v = self._compute_tube_radius()

        # 3. 判断是否触发求解
        triggered, trigger_error_norm = self._check_trigger(x, omega_s, omega_v)

        # 4. 求解MPC或复用上一帧序列
        if triggered:
            u_bar_seq, x_bar_seq = self._solve_mpc(x, desired_gap, omega_s, omega_v)
            u_bar_used = float(u_bar_seq[0])
            x_bar_used = x_bar_seq[0]

            self.u_bar_seq_last = u_bar_seq
            self.x_bar_seq_last = x_bar_seq
            self.trigger_hold_index = 1
        else:
            u_bar_used = float(self.u_bar_seq_last[self.trigger_hold_index])
            x_bar_used = self.x_bar_seq_last[self.trigger_hold_index]
            self.trigger_hold_index += 1

        # 5. Tube反馈修正
        z0 = x - x_bar_used
        u_feedback = float(self.K @ z0)
        a_cmd = u_bar_used + u_feedback
        a_cmd = float(np.clip(a_cmd, self.a_min, self.a_max))

        # 6. 更新历史
        self.u_prev = a_cmd
        
        debug_info = {
            "tau": tau,
            "h": h,
            "desired_gap": desired_gap,
            "gap": gap,
            "e_s": e_s,
            "e_v": e_v,
            "omega_s": omega_s,
            "omega_v": omega_v,
            "triggered": triggered,
            "trigger_error_norm": trigger_error_norm,
            "u_bar": u_bar_used,
            "u_feedback": u_feedback,
            "a_cmd": a_cmd,
        }
        return a_cmd, debug_info

    def _compute_error(self):
        """
        计算 DTH 期望间距和纵向误差状态。

        e_s : 实际间距 - 期望间距
        e_v : 前车速度 - 自车速度
        """
        h = self.h0 + self.gamma * self.tau
        h = float(np.clip(h, self.h_min, self.h_max))

        desired_gap = h * self.vehicle_state["speed"] + self.d0
        gap = self.front_state["s"] - self.vehicle_state["s"]

        e_s = gap - desired_gap
        e_v = self.front_state["speed"] - self.vehicle_state["speed"]
        x = np.array([e_s, e_v])

        return x, h, desired_gap, gap, e_s, e_v

    def _compute_tube_radius(self):
        """
        根据通信延迟计算自适应 Tube 半径。
        """
        omega_s = self.omega_s0 + self.beta_s * self.tau
        omega_v = self.omega_v0 + self.beta_v * self.tau

        omega_s = float(np.clip(omega_s, self.omega_s_min, self.omega_s_max))
        omega_v = float(np.clip(omega_v, self.omega_v_min, self.omega_v_max))

        return omega_s, omega_v

    def _check_trigger(self, x, omega_s, omega_v):
        """
        判断本周期是否需要重新求解 MPC。
        """
        trigger_error_norm = 0.0

        # 首次运行或无可复用序列
        if self.u_bar_seq_last is None or self.x_bar_seq_last is None:
            return True, trigger_error_norm

        # 上一轮序列已经用完
        if self.trigger_hold_index >= len(self.u_bar_seq_last):
            return True, trigger_error_norm

        # 通信延迟超过阈值
        if self.tau > self.tau_thresh:
            return True, trigger_error_norm

        # Tube 归一化偏差超过阈值
        x_bar_ref = self.x_bar_seq_last[self.trigger_hold_index]
        z = x - x_bar_ref
        trigger_error_norm = np.linalg.norm([
            z[0] / omega_s,
            z[1] / omega_v,
        ])

        triggered = trigger_error_norm > self.sigma
        return triggered, float(trigger_error_norm)

    def _solve_mpc(self, x0, desired_gap, omega_s, omega_v):

        # 更新初始状态和模型矩阵
        A, B = self._build_model()
        self.x0_param.value          = x0
        self.A_param.value           = A
        self.B_param.value           = B
        self.u_prev_param.value      = self.u_prev
        self.desired_gap_param.value = desired_gap
        self.front_speed_param.value = self.front_state["speed"]
        self.omega_s_param.value     = omega_s
        self.omega_v_param.value     = omega_v

        # 求解
        try:
            self.mpc_problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except cp.error.SolverError:
            print("[FollowerLonMPC] 求解异常，保持上一帧")
            return self._fallback_seq(x0)

        # 保底
        if self.u_var.value is None or self.x_var.value is None:
            print("[FollowerLonMPC] 求解失败，保持上一帧")
            return self._fallback_seq(x0)

        u_bar_seq = np.array(self.u_var.value).reshape(-1)
        x_bar_seq = np.array(self.x_var.value).T

        return u_bar_seq, x_bar_seq

    def _fallback_seq(self, x0):
        u_safe = float(np.clip(self.u_prev, self.a_min, self.a_max))
        u_bar_seq = np.full(self.N, u_safe)
        x_bar_seq = np.tile(x0, (self.N + 1, 1))
        return u_bar_seq, x_bar_seq

    def _build_model(self):
        A = np.array([[1.0, self.dt],
                    [0.0, 1.0    ]])
        B = np.array([[0.0       ],
                    [-self.dt    ]])
        return A, B

    def _setup_mpc(self):

        # 1: 声明优化变量
        self.x_var = cp.Variable((2, self.N + 1))   # 状态序列 [e_s, e_v]
        self.u_var = cp.Variable((1, self.N))        # 控制序列 [a_ego]

        # 2: 声明每帧更新的参数
        self.x0_param          = cp.Parameter(2)                # 初始误差状态
        self.A_param           = cp.Parameter((2, 2))           # 状态矩阵
        self.B_param           = cp.Parameter((2, 1))           # 控制矩阵
        self.u_prev_param      = cp.Parameter()                 # 上一帧加速度
        self.desired_gap_param = cp.Parameter(nonneg=True)      # DTH期望间距
        self.front_speed_param = cp.Parameter(nonneg=True)      # 前车速度
        self.omega_s_param     = cp.Parameter(nonneg=True)      # 间距Tube半径
        self.omega_v_param     = cp.Parameter(nonneg=True)      # 速度Tube半径

        # 3: 构建代价函数和约束
        cost        = 0
        constraints = [self.x_var[:, 0] == self.x0_param]

        for k in range(self.N):
            # 加速度增量（k=0时参考上一帧，k>0时参考上一步）
            u_ref = self.u_prev_param if k == 0 else self.u_var[0, k - 1]
            du    = self.u_var[0, k] - u_ref

            gap_bar   = self.x_var[0, k] + self.desired_gap_param
            v_ego_bar = self.front_speed_param - self.x_var[1, k]

            # 代价：间距误差 + 速度误差 + 加速度幅值 + 加速度增量
            cost += cp.quad_form(self.x_var[:, k], self.Q)
            cost += self.R_u  * cp.square(self.u_var[0, k])
            cost += self.R_du * cp.square(du)

            # 约束：动力学 + 加速度幅值
            constraints += [self.x_var[:, k + 1] == self.A_param @ self.x_var[:, k] + self.B_param @ self.u_var[:, k]]
            constraints += [self.u_var[0, k] >= self.a_min, self.u_var[0, k] <= self.a_max]

            # 收紧约束：间距 + 自车速度
            constraints += [gap_bar >= self.gap_min + self.omega_s_param, gap_bar <= self.gap_max - self.omega_s_param]
            constraints += [v_ego_bar >= self.v_min + self.omega_v_param, v_ego_bar <= self.v_max - self.omega_v_param]

        # 终端代价
        cost += cp.quad_form(self.x_var[:, self.N], self.Q)

        # 4: 构建问题
        self.mpc_problem = cp.Problem(cp.Minimize(cost), constraints)
