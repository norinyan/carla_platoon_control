# -*- coding: utf-8 -*-

import numpy as np
import cvxpy as cp
from math import sin, cos, atan2, radians
from .base_controller import PlatoonController


class _LatLQR:
    def __init__(self, params):
        self.L             = params.get("wheelbase",    2.85)
        self.max_steer_rad = radians(params.get("max_steer_deg", 35.0))

        # Riccati 迭代参数
        self.max_iter = params.get("max_iter", 100)
        self.eps      = params.get("eps",      1e-4)

        # 状态代价矩阵 Q：惩罚 e_y 和 e_yaw
        q_ey   = params.get("q_ey",   1.0)
        q_eyaw = params.get("q_eyaw", 1.0)
        self.Q = np.diag([q_ey, q_eyaw])

        # 控制代价矩阵 R：惩罚转角幅值
        r_delta = params.get("r_delta", 1.0)
        self.R  = np.array([[r_delta]])

    def compute(self, vehicle_state, ref_points, dt):
        """
        vehicle_state : {x, y, yaw, speed}
        ref_points    : 从最近点开始往前的 N 个参考点，每点含 {x, y, yaw, kappa, speed, s}
        dt            : 控制周期 (s)
        return        : steer，范围 [-1.0, 1.0]
        """
        self.dt            = dt
        self.vehicle_state = vehicle_state
        self.ref_points    = ref_points

        # 1. 计算误差状态
        e_y, e_yaw = self._compute_error()

        # 2. 构建模型
        A, B = self._build_model()

        # 3. 求解LQR增益
        K = self._solve_lqr(A, B)

        # 4. 计算最优前轮转角
        x = np.array([e_y, e_yaw])
        delta_opt = -K @ x
        delta_opt = float(delta_opt)

        # 5. 输出steer
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
        x, y = self.vehicle_state["x"], self.vehicle_state["y"]
        yaw  = self.vehicle_state["yaw"]
        ref  = self.ref_points[0]
        x_ref, y_ref = ref["x"], ref["y"]
        yaw_ref      = ref["yaw"]

        dx  = x - x_ref
        dy  = y - y_ref
        e_y = -dx * sin(yaw_ref) + dy * cos(yaw_ref)

        e_yaw = yaw - yaw_ref
        e_yaw = atan2(sin(e_yaw), cos(e_yaw))

        return e_y, e_yaw

    def _solve_lqr(self, A, B):
        # 初始化 P 为状态代价矩阵 Q
        P = self.Q.copy()

        for _ in range(self.max_iter):
            # P_new = Q + A'PA - A'PB · (R + B'PB)^{-1} · B'PA
            P_new = (self.Q
                     + A.T @ P @ A
                     - A.T @ P @ B @ np.linalg.pinv(self.R + B.T @ P @ B) @ B.T @ P @ A)

            # 收敛判断：P 的最大元素变化量小于阈值则退出
            if abs(P_new - P).max() < self.eps:
                break
            P = P_new

        # K = (R + B'PB)^{-1} · B'PA
        K = np.linalg.pinv(self.R + B.T @ P @ B) @ B.T @ P @ A
        return K

    def _build_model(self):
        v = max(self.vehicle_state["speed"], 0.5)
        A = np.array([[1.0, v * self.dt],
                      [0.0, 1.0       ]])
        B = np.array([[0.0                  ],
                      [- v * self.dt / self.L]])
        return A, B


class PlatoonMPCController(PlatoonController):
    def __init__(self, params):
        # 横向LQR
        self.lat_ctrl = _LatLQR(params['lat_lqr'])

        # 车辆参数
        self.max_steer_rad = radians(params['lat_lqr'].get('max_steer_deg', 35.0))
        self.vehicle_length = params.get('vehicle_length', 4.9)

        # 纵向MPC：领航车参数
        p_l = params['lon_mpc_leader']
        self.leader_N     = p_l.get('N',     10)
        self.leader_q_ev  = p_l.get('q_ev',  5.0)
        self.leader_R_a   = p_l.get('R_a',   0.1)
        self.leader_a_min = p_l.get('a_min', -3.0)
        self.leader_a_max = p_l.get('a_max',  1.5)

        # 纵向MPC：跟随车参数
        p_f = params['lon_mpc_follower']
        self.follower_N     = p_f.get('N',     10)
        self.follower_q_es  = p_f.get('q_es',  1.5)
        self.follower_q_ev  = p_f.get('q_ev',  0.8)
        self.follower_R_a   = p_f.get('R_a',   0.75)
        self.follower_a_min = p_f.get('a_min', -3.0)
        self.follower_a_max = p_f.get('a_max',  1.5)

        # 控制周期
        self.Ts = params.get('ctrl_period', 0.1)

        # 标定表（由外部注入，_map2cmd使用）
        self.calib_speed_bins    = []
        self.calib_throttle_list = []
        self.calib_grid          = []
        self.BRAKE_K             = 0.2

        # 构建MPC问题（延迟到第一次调用）
        self._leader_mpc   = None
        self._follower_mpc = None

    # ─────────────────────────────────────────────
    #  对外接口
    # ─────────────────────────────────────────────
    def compute(self, ego_state, ref_points, leader_state, desired_gap, dt):
        """
        ego_state   : {x, y, yaw, speed}
        ref_points  : 从最近点开始往前N个参考点，每点含 {x, y, yaw, kappa, speed, s}
        leader_state: 前车状态 {x, y, yaw, speed, s}，领航车传 None
        desired_gap : 期望净跟车间距(m)，按前后车中心距减去车长计算
        dt          : 控制周期(s)
        return      : (steer, a_cmd)
        """
        self._v_now = ego_state['speed']

        # 横向LQR
        steer = self.lat_ctrl.compute(ego_state, ref_points, dt)

        # 纵向MPC
        if leader_state is None:
            # 领航车：跟参考速度
            e_v   = ref_points[0]['speed'] - ego_state['speed']
            a_cmd = self._solve_leader_mpc(e_v)
        else:
            # 跟随车：跟间距+速度
            ego_s    = ref_points[0]['s']
            e_s      = (leader_state['s'] - ego_s - self.vehicle_length) - desired_gap
            e_v      = leader_state['speed'] - ego_state['speed']
            a_cmd    = self._solve_follower_mpc(e_s, e_v)

        return steer, a_cmd

    def map2cmd(self, a_cmd, steer):
        """a_cmd → {throttle, brake, steer}，复用单车标定表逻辑"""
        dead_zone = 0.05
        v_now     = 0.0  # 由外部在调用前更新 self._v_now
        v_now     = getattr(self, '_v_now', 0.0)

        if a_cmd >= 0.0:
            throttle_cmd  = self._interp_2d(a_cmd, v_now)
            throttle_cmd += 0.0153 * v_now
            brake_cmd     = 0.0
        else:
            if abs(a_cmd) < dead_zone:
                throttle_cmd = 0.0
                brake_cmd    = 0.0
            else:
                throttle_cmd = 0.0
                brake_cmd    = max(0.0, min(1.0, abs(a_cmd) * self.BRAKE_K))

        return {
            'throttle': max(0.0, min(1.0, throttle_cmd)),
            'brake':    max(0.0, min(1.0, brake_cmd)),
            'steer':    max(-1.0, min(1.0, steer)),
        }

    # ─────────────────────────────────────────────
    #  领航车MPC：单状态 [e_v]
    # ─────────────────────────────────────────────
    def _build_leader_mpc(self):
        N     = self.leader_N
        Ts    = self.Ts
        q_ev  = self.leader_q_ev
        R_a   = self.leader_R_a
        a_min = self.leader_a_min
        a_max = self.leader_a_max

        # 系统: e_v(k+1) = e_v(k) - Ts * u(k)
        A = np.array([[1.0]])
        B = np.array([[-Ts]])
        Q = np.array([[q_ev]])
        R = np.array([[R_a]])
        P = Q

        x_var    = cp.Variable((1, N + 1))
        u_var    = cp.Variable(N)
        x0_param = cp.Parameter(1)

        cost        = 0
        constraints = [x_var[:, 0] == x0_param]
        for k in range(N):
            constraints += [
                x_var[:, k + 1] == A @ x_var[:, k] + B.flatten() * u_var[k],
                a_min <= u_var[k],
                u_var[k] <= a_max,
            ]
            cost += cp.quad_form(x_var[:, k], Q) + R[0, 0] * cp.square(u_var[k])
        cost += cp.quad_form(x_var[:, N], P)

        problem = cp.Problem(cp.Minimize(cost), constraints)
        self._leader_mpc = {
            'problem':  problem,
            'x_var':    x_var,
            'u_var':    u_var,
            'x0_param': x0_param,
        }

    def _solve_leader_mpc(self, e_v):
        if self._leader_mpc is None:
            self._build_leader_mpc()

        mpc = self._leader_mpc
        mpc['x0_param'].value = np.array([e_v], dtype=float)

        try:
            mpc['problem'].solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except cp.error.SolverError:
            return 0.0

        if mpc['u_var'].value is None:
            return 0.0

        return float(mpc['u_var'].value[0])

    # ─────────────────────────────────────────────
    #  跟随车MPC：双状态 [e_s, e_v]
    # ─────────────────────────────────────────────
    def _build_follower_mpc(self):
        N     = self.follower_N
        Ts    = self.Ts
        q_es  = self.follower_q_es
        q_ev  = self.follower_q_ev
        R_a   = self.follower_R_a
        a_min = self.follower_a_min
        a_max = self.follower_a_max

        # 系统: 参考pid.py setup_mpc
        # e_s(k+1) = e_s(k) + Ts * e_v(k)
        # e_v(k+1) = e_v(k) - Ts * u(k)
        A = np.array([[1.0, Ts ],
                      [0.0, 1.0]])
        B = np.array([[0.0 ],
                      [-Ts ]])
        Q = np.diag([q_es, q_ev])
        R = np.array([[R_a]])
        P = Q

        x_var    = cp.Variable((2, N + 1))
        u_var    = cp.Variable(N)
        x0_param = cp.Parameter(2)

        cost        = 0
        constraints = [x_var[:, 0] == x0_param]
        for k in range(N):
            constraints += [
                x_var[:, k + 1] == A @ x_var[:, k] + B.flatten() * u_var[k],
                a_min <= u_var[k],
                u_var[k] <= a_max,
            ]
            cost += cp.quad_form(x_var[:, k], Q) + R[0, 0] * cp.square(u_var[k])
        cost += cp.quad_form(x_var[:, N], P)

        problem = cp.Problem(cp.Minimize(cost), constraints)
        self._follower_mpc = {
            'problem':  problem,
            'x_var':    x_var,
            'u_var':    u_var,
            'x0_param': x0_param,
        }

    def _solve_follower_mpc(self, e_s, e_v):
        if self._follower_mpc is None:
            self._build_follower_mpc()

        mpc = self._follower_mpc
        mpc['x0_param'].value = np.array([e_s, e_v], dtype=float)

        try:
            mpc['problem'].solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except cp.error.SolverError:
            return 0.0

        if mpc['u_var'].value is None:
            return 0.0

        return float(mpc['u_var'].value[0])

    # ─────────────────────────────────────────────
    #  标定表查表（复用单车逻辑）
    # ─────────────────────────────────────────────
    def _interp_2d(self, a_des, v_now):
        a_at_v = []
        for row in self.calib_grid:
            valid = [
                (self.calib_speed_bins[j], row[j])
                for j in range(len(self.calib_speed_bins))
                if row[j] is not None
            ]
            if len(valid) < 2:
                a_at_v.append(None)
                continue
            if v_now <= valid[0][0]:
                a_at_v.append(valid[0][1])
                continue
            if v_now >= valid[-1][0]:
                a_at_v.append(valid[-1][1])
                continue
            val = None
            for k in range(len(valid) - 1):
                x0, y0 = valid[k]
                x1, y1 = valid[k + 1]
                if x0 <= v_now <= x1:
                    val = y0 + (y1 - y0) * (v_now - x0) / (x1 - x0)
                    break
            a_at_v.append(val)

        pairs = [
            (a_at_v[i], self.calib_throttle_list[i])
            for i in range(len(self.calib_throttle_list))
            if a_at_v[i] is not None
        ]
        if len(pairs) < 2:
            return 0.0

        pairs.sort(key=lambda x: x[0])

        if a_des <= pairs[0][0]:
            return pairs[0][1]
        if a_des >= pairs[-1][0]:
            return pairs[-1][1]

        for i in range(len(pairs) - 1):
            a0, t0 = pairs[i]
            a1, t1 = pairs[i + 1]
            if a0 <= a_des <= a1:
                return max(0.0, min(1.0, t0 + (t1 - t0) * (a_des - a0) / (a1 - a0)))

        return 0.0
