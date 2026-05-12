# 修改规划 -May11 第二版

## 1. 这个想法可以实现吗，怎么做

可以实现。

做法：不改源码控制逻辑，只做三套参数配置 + 三组实验 CSV 对比。

三套版本：

```text
V1: Base-NoDelay-NoTrigger
V2: Base-Delay-NoTrigger
V3: Proposed-Delay-DTH-AdaptiveTube-Trigger
```

实验方法：

```text
1. 每次只换 controller_params.yaml 中 follower_lon_mpc 的参数。
2. 每个版本跑同一条 route A。
3. 每个版本记录 platoon_control_route_A_*.csv。
4. 离线统计 e_s RMSE、e_v RMSE、min gap、trigger rate、solve time、accel saturation。
5. 用 V1 证明基础控制能跑。
6. 用 V2 证明加入通信延迟后性能变差。
7. 用 V3 证明三项设计能恢复或改善性能。
```

版本命名中的 A/B/C 建议定义为：

```text
A = Delay Model
    NoDelay / Delay

B = Spacing + Tube Design
    FixedCTH-FixedTube / DTH-AdaptiveTube

C = Solver Trigger Mode
    Periodic / EventTrigger
```

因此三版命名：

```text
V1 = NoDelay + FixedCTH-FixedTube + Periodic
V2 = Delay + FixedCTH-FixedTube + Periodic
V3 = Delay + DTH-AdaptiveTube + EventTrigger
```

## 2. 三个版本区别和参数怎么调

### V1: 无通信延迟 Base

目的：

```text
验证基础编队控制本身可以稳定工作。
```

配置：

```yaml
follower_lon_mpc:
  h0: 1.0
  gamma: 0.0
  h_min: 1.0
  h_max: 1.0
  d0: 5.0

  omega_s0: 0.5
  omega_v0: 0.2
  beta_s: 0.0
  beta_v: 0.0
  omega_s_min: 0.5
  omega_s_max: 0.5
  omega_v_min: 0.2
  omega_v_max: 0.2

  sigma: -1.0
  tau_thresh: 999.0
```

含义：

```text
gamma=0       -> 期望间距不随 tau 变化
beta_s/v=0   -> Tube 半径不随 tau 变化
sigma=-1     -> 每周期触发，相当于 Periodic MPC
tau_thresh=999 -> 延迟条件永远不触发
```

运行时：

```text
tau 仍可由代码生成，但参数让 tau 不影响控制器。
```

### V2: 加通信延迟 Base

目的：

```text
验证通信延迟存在时，固定间距 + 固定 Tube + 周期 MPC 的控制效果变差。
```

配置：

```yaml
follower_lon_mpc:
  h0: 1.0
  gamma: 0.0
  h_min: 1.0
  h_max: 1.0
  d0: 5.0

  omega_s0: 0.5
  omega_v0: 0.2
  beta_s: 0.0
  beta_v: 0.0
  omega_s_min: 0.5
  omega_s_max: 0.5
  omega_v_min: 0.2
  omega_v_max: 0.2

  sigma: -1.0
  tau_thresh: 999.0
```

和 V1 参数一样。

区别：

```text
V1: 不注入或不使用延迟影响。
V2: 实验场景中加入通信延迟影响。
```

如果当前源码只是把 tau 用于 DTH/Tube/Trigger，没有真正延迟前车状态，则 V2 和 V1 不会明显拉开。

这种情况下需要后续增加 delayed front_state buffer，才是真正的 Delay Base。

### V3: 加通信延迟 + 三项设计

目的：

```text
验证 DTH + 自适应 Tube + 事件触发 能改善延迟下的控制效果。
```

配置：

```yaml
follower_lon_mpc:
  h0: 1.0
  gamma: 3.3
  h_min: 0.6
  h_max: 1.5
  d0: 5.0

  omega_s0: 0.5
  omega_v0: 0.2
  beta_s: 3.0
  beta_v: 1.0
  omega_s_min: 0.3
  omega_s_max: 3.0
  omega_v_min: 0.1
  omega_v_max: 1.5

  sigma: 0.8
  tau_thresh: 0.07
```

含义：

```text
gamma > 0       -> DTH 动态时距
beta_s/v > 0    -> Tube 半径随 tau 自适应
sigma=0.8       -> Tube 越界触发
tau_thresh=0.07 -> 高延迟触发
```

## 3. Base 应该包含什么，我的创新是在 Base 上加了什么

Base 应该包含：

```text
1. 三车 PF 编队拓扑：0 -> 1 -> 2
2. Leader 纵向 PID 或速度跟踪控制
3. Follower 纵向 MPC
4. 固定时距 CTH
5. 固定 Tube 半径
6. 周期性 MPC 求解
7. 车辆加速度、速度、间距物理约束
8. 同一路径 s 坐标下的 gap 和误差计算
```

Base 不应该包含：

```text
1. tau 驱动的动态时距
2. tau 驱动的 Tube 半径变化
3. 事件触发求解
4. 延迟阈值触发
```

你的创新是在 Base 上增加：

```text
创新 1：DTH
固定 h -> h = clip(h0 + gamma * tau, h_min, h_max)
固定 desired_gap -> desired_gap = h(tau) * v_ego + d0

创新 2：Adaptive Tube
固定 omega_s / omega_v
-> omega_s = clip(omega_s0 + beta_s * tau, ...)
-> omega_v = clip(omega_v0 + beta_v * tau, ...)

创新 3：Event Trigger
每周期求解 MPC
-> 当 Tube 偏差超过阈值或 tau 超过阈值时才重新求解
```

最终论文对比逻辑：

```text
V1 证明 Base 本身有效。
V2 证明 Base 在通信延迟下性能下降。
V3 证明你的三个设计能提升延迟场景下的鲁棒性、安全性或资源效率。
```

注意：

```text
如果 V2 要真正体现“通信延迟导致性能变差”，必须让 follower 使用延迟后的 front_state。
如果只是随机生成 tau，但 front_state 仍是当前前车状态，则 tau 没有真正破坏信息时效性，V2 不一定会比 V1 差。
```















# DTE 编队控制 MVP 修改规划  -May11  第一版

目标：在现有 CARLA + ROS2 三车编队骨架上，实现最小可运行版本的三个创新：

```text
创新 1：基于通信延迟 tau 的动态时距 DTH
创新 2：基于通信延迟 tau 的自适应 TubeMPC
创新 3：事件触发与 TubeMPC 协同
```

MVP 原则：

- 只做能支撑三项创新的最小闭环。
- 不加入复杂增强、额外测试框架、过度保护逻辑。
- 先使用模拟通信延迟 `tau`，CARLA 本身不提供 V2X。
- 控制器只做算法，ROS 数据流放在 `platoon_node.py`。

## 1. 修改文件

```text
carla_platoon_control/controllers/platoon_controller.py
carla_platoon_control/platoon_node.py
carla_platoon_control/config/controller_params.yaml
```

`visualizer.py` 暂不强制修改。先通过 `debug_info` 和日志/CSV 记录关键数据。

## 2. 实现顺序

### 第一步：补齐 controller_params.yaml

在 `follower_lon_mpc` 下加入参数：

```yaml
follower_lon_mpc:
  N: 10
  dt: 0.1

  h0: 0.8
  gamma: 3.3
  h_min: 0.6
  h_max: 1.5
  d0: 1.5

  q_s: 4.0
  q_v: 1.0
  r_u: 0.2
  r_du: 0.5

  a_min: -2.0
  a_max: 1.5
  v_min: 0.0
  v_max: 8.0
  gap_min: 1.0
  gap_max: 30.0

  omega_s0: 0.5
  omega_v0: 0.2
  beta_s: 3.0
  beta_v: 1.0
  omega_s_min: 0.3
  omega_s_max: 3.0
  omega_v_min: 0.1
  omega_v_max: 1.5

  k_s: 0.2
  k_v: 0.6

  sigma: 0.8
```

## 3. platoon_node.py 修改

### 3.1 模拟通信延迟

在 `PlatoonControlNode` 中增加：

```python
def _get_v2x_delay(self, front_id, rear_id):
    return random.uniform(0.01, 0.10)
```

单位必须是秒：

```text
50 ms -> 0.05
```

### 3.2 调用 follower 控制器时传入 tau

在 `_on_timer()` 中，三辆车都完成 `_find_ref_points()` 后，再计算：

```python
tau_01 = self._get_v2x_delay("vehicle_0", "vehicle_1")
tau_12 = self._get_v2x_delay("vehicle_1", "vehicle_2")
```

调用改为：

```python
accel_1, info_1 = self.follower_lon_ctrl_1.compute(
    state_1,
    ref_points_1,
    state_0,
    tau_01,
    self.ctrl_period,
)

accel_2, info_2 = self.follower_lon_ctrl_2.compute(
    state_2,
    ref_points_2,
    state_1,
    tau_12,
    self.ctrl_period,
)
```

MVP 中 `front_state` 先使用当前前车状态。`tau` 先用于 DTH 和自适应 Tube。

## 4. FollowerLonMPC 接口

当前：

```python
def compute(self, vehicle_state, ref_points, front_state, desired_gap, dt):
    a_cmd = 0.0
    return a_cmd
```

改为：

```python
def compute(self, vehicle_state, ref_points, front_state, tau, dt):
    ...
    return a_cmd, debug_info
```

含义：

```text
vehicle_state：当前跟随车自车状态
front_state：前车状态
tau：前车到当前跟随车的单向通信延迟
dt：控制周期
```

控制器内部不生成 `tau`，不修改 `vehicle_state` 和 `front_state` 原字典。

## 5. DTH 动态时距

在 `compute()` 内部计算：

```python
h = clip(h0 + gamma * tau, h_min, h_max)
desired_gap = h * vehicle_state["speed"] + d0
gap = front_state["s"] - vehicle_state["s"]
e_s = gap - desired_gap
e_v = front_state["speed"] - vehicle_state["speed"]
x = [e_s, e_v]^T
```

要求：

```text
gap 使用路径坐标 s
vehicle_state["speed"] 是当前跟随车速度
front_state["s"] 和 vehicle_state["s"] 必须来自同一条参考轨迹
```

## 6. 自适应 TubeMPC

### 6.1 误差状态

```text
x = [e_s, e_v]^T
e_s = gap - desired_gap
e_v = v_front - v_ego
u = a_ego
```

MVP 预测模型使用当前 `desired_gap`，在预测时域内保持 `h` 和 `desired_gap` 不变：

```text
x_bar(j+1) = A x_bar(j) + B u_bar(j) + E a_front
```

其中：

```text
A = [[1, dt],
     [0, 1 ]]

B = [[0 ],
     [-dt]]

E = [[0 ],
     [dt]]
```

前车加速度 MVP 取：

```text
a_front = 0
```

### 6.2 通信延迟自适应 Tube 半径

```python
omega_s = clip(omega_s0 + beta_s * tau, omega_s_min, omega_s_max)
omega_v = clip(omega_v0 + beta_v * tau, omega_v_min, omega_v_max)
```

含义：

```text
tau 增大 -> omega_s / omega_v 增大 -> Tube 扩张 -> 约束更保守
tau 减小 -> omega_s / omega_v 减小 -> Tube 收缩 -> 保守性降低
```

### 6.3 标称约束收紧

标称间距：

```text
gap_bar(j) = e_s_bar(j) + desired_gap
```

标称自车速度：

```text
v_ego_bar(j) = front_state["speed"] - e_v_bar(j)
```

收紧约束：

```text
gap_min + omega_s <= gap_bar(j) <= gap_max - omega_s
v_min + omega_v <= v_ego_bar(j) <= v_max - omega_v
a_min <= u_bar(j) <= a_max
```

### 6.4 目标函数

```text
J = Σ [
  q_s * e_s_bar(j)^2
+ q_v * e_v_bar(j)^2
+ r_u * u_bar(j)^2
+ r_du * (u_bar(j) - u_bar(j-1))^2
]
```

`j=0` 时：

```text
u_bar(-1) = u_prev
```

`u_prev` 为上一周期实际输出的 `a_cmd`，初始化为 `0.0`。

### 6.5 Tube 反馈修正

MPC 求解得到：

```text
u_bar_seq
x_bar_seq
```

取第一步：

```python
u_bar_0 = u_bar_seq[0]
x_bar_0 = x_bar_seq[0]
z0 = x - x_bar_0
u = u_bar_0 + K @ z0
a_cmd = clip(u, a_min, a_max)
```

其中：

```text
K = [k_s, k_v]
```

在当前误差定义：

```text
e_s = gap - desired_gap
e_v = v_front - v_ego
```

下，`k_s > 0`、`k_v > 0`。

## 7. 事件触发

控制器保存：

```text
u_prev = 0.0
u_bar_seq_last = None
x_bar_seq_last = None
trigger_hold_index = 1
```

每周期先判断是否强制触发：

```python
if self.u_bar_seq_last is None or self.x_bar_seq_last is None:
    triggered = True
elif self.trigger_hold_index >= len(self.u_bar_seq_last):
    triggered = True
else:
    x_bar_ref = self.x_bar_seq_last[self.trigger_hold_index]
    z = x - x_bar_ref
    trigger_error_norm = norm([z[0] / omega_s, z[1] / omega_v])
    triggered = trigger_error_norm > sigma
```

触发时：

```python
u_bar_seq, x_bar_seq = solve_mpc(...)
u_bar_0 = u_bar_seq[0]
x_bar_0 = x_bar_seq[0]
z0 = x - x_bar_0
a_cmd = clip(u_bar_0 + K @ z0, a_min, a_max)

self.u_bar_seq_last = u_bar_seq
self.x_bar_seq_last = x_bar_seq
self.trigger_hold_index = 1
```

不触发时：

```python
u_bar_hold = self.u_bar_seq_last[self.trigger_hold_index]
x_bar_hold = self.x_bar_seq_last[self.trigger_hold_index]
z0 = x - x_bar_hold
a_cmd = clip(u_bar_hold + K @ z0, a_min, a_max)
self.trigger_hold_index += 1
```

每次输出后保存：

```python
self.u_prev = a_cmd
```

## 8. debug_info

`compute()` 返回：

```python
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
    "u_feedback": float(K @ z0),
    "a_cmd": a_cmd,
}
```

## 9. 最小验收

实现完成后只检查以下结果：

```text
1. platoon_node.py 能正常调用 follower，并解包 accel, info。
2. vehicle_1 / vehicle_2 的 accel 不再恒为 0。
3. tau 增大时 desired_gap 增大。
4. tau 增大时 omega_s / omega_v 增大。
5. triggered 能在 True / False 之间变化。
6. a_cmd 能通过现有 _map2cmd() 转成 throttle / brake。
```

## 10. 第三步 FollowerLonMPC 实现细化

第三步只修改：

```text
carla_platoon_control/controllers/platoon_controller.py
```

不修改：

```text
platoon_node.py
base_controller.py
controller_params.yaml
visualizer.py
```

目标：

```text
先实现 DTH + 自适应 TubeMPC 的每周期求解版本。
事件触发的序列复用逻辑留到第四步。
```

因此第三步中：

```text
triggered 固定为 True
trigger_error_norm 固定为 0.0
每个控制周期都重新求解一次 QP
```

### 10.1 代码结构

`FollowerLonMPC` 按模板 `lat_mpc.py` 的风格组织为：

```text
1. __init__()
2. compute()
3. _compute_error()
4. _compute_tube_radius()
5. _solve_mpc()
6. _build_model()
7. _setup_mpc()
```

注释风格沿用现有代码：

```text
# 1. 计算误差状态
# 2. 求解MPC输出
# 3. 更新历史
```

不新增复杂抽象，不拆分新文件。

### 10.2 __init__()

读取参数：

```text
N, dt

h0, gamma, h_min, h_max, d0

q_s, q_v, r_u, r_du

a_min, a_max
v_min, v_max
gap_min, gap_max

omega_s0, omega_v0
beta_s, beta_v
omega_s_min, omega_s_max
omega_v_min, omega_v_max

k_s, k_v

sigma, tau_thresh
```

内部矩阵：

```python
self.Q = np.diag([q_s, q_v])
self.R_u = r_u
self.R_du = r_du
self.K = np.array([k_s, k_v])
```

历史量：

```python
self.u_prev = 0.0
self.u_bar_seq_last = None
self.x_bar_seq_last = None
self.trigger_hold_index = 1
```

说明：

```text
事件触发相关历史量第三步先初始化但不使用。
第四步再接入序列复用。
```

最后调用：

```python
self._setup_mpc()
```

### 10.3 compute()

接口：

```python
def compute(self, vehicle_state, ref_points, front_state, tau, dt):
```

主流程：

```text
1. 保存当前输入到 self
2. 计算误差状态
3. 计算 Tube 半径
4. 求解 MPC
5. Tube 反馈修正
6. 更新 u_prev
7. 返回 a_cmd, debug_info
```

保存输入：

```python
self.dt = dt
self.vehicle_state = vehicle_state
self.ref_points = ref_points
self.front_state = front_state
self.tau = tau
```

不修改原始输入字典：

```text
compute() 内部只读取 vehicle_state 和 front_state。
不向两个字典写入新字段。
```

### 10.4 _compute_error()

计算 DTH：

```python
h = np.clip(self.h0 + self.gamma * self.tau, self.h_min, self.h_max)
desired_gap = h * self.vehicle_state["speed"] + self.d0
```

计算误差：

```python
gap = self.front_state["s"] - self.vehicle_state["s"]
e_s = gap - desired_gap
e_v = self.front_state["speed"] - self.vehicle_state["speed"]
x = np.array([e_s, e_v])
```

返回：

```python
return x, h, desired_gap, gap, e_s, e_v
```

要求：

```text
gap 使用路径坐标 s。
front_state["s"] 和 vehicle_state["s"] 必须来自同一条参考轨迹。
```

### 10.5 _compute_tube_radius()

计算：

```python
omega_s = np.clip(
    self.omega_s0 + self.beta_s * self.tau,
    self.omega_s_min,
    self.omega_s_max,
)

omega_v = np.clip(
    self.omega_v0 + self.beta_v * self.tau,
    self.omega_v_min,
    self.omega_v_max,
)
```

返回：

```python
return omega_s, omega_v
```

含义：

```text
tau 增大 -> omega_s / omega_v 增大 -> 收紧约束更保守
tau 减小 -> omega_s / omega_v 减小 -> 收紧约束更宽松
```

### 10.6 _build_model()

MVP 使用二阶误差模型：

```python
A = np.array([
    [1.0, self.dt],
    [0.0, 1.0],
])

B = np.array([
    [0.0],
    [-self.dt],
])
```

第三步中：

```text
a_front = 0
E a_front 暂不进入模型
```

返回：

```python
return A, B
```

### 10.7 _setup_mpc()

变量：

```python
self.x_var = cp.Variable((2, self.N + 1))
self.u_var = cp.Variable((1, self.N))
```

参数：

```python
self.x0_param = cp.Parameter(2)
self.A_param = cp.Parameter((2, 2))
self.B_param = cp.Parameter((2, 1))
self.u_prev_param = cp.Parameter()
self.desired_gap_param = cp.Parameter(nonneg=True)
self.front_speed_param = cp.Parameter(nonneg=True)
self.omega_s_param = cp.Parameter(nonneg=True)
self.omega_v_param = cp.Parameter(nonneg=True)
```

初始约束：

```python
constraints = [self.x_var[:, 0] == self.x0_param]
```

每个预测步：

```python
u_ref = self.u_prev_param if k == 0 else self.u_var[0, k - 1]
du = self.u_var[0, k] - u_ref

gap_bar = self.x_var[0, k] + self.desired_gap_param
v_ego_bar = self.front_speed_param - self.x_var[1, k]
```

代价：

```python
cost += cp.quad_form(self.x_var[:, k], self.Q)
cost += self.R_u * cp.square(self.u_var[0, k])
cost += self.R_du * cp.square(du)
```

动力学约束：

```python
constraints += [
    self.x_var[:, k + 1] ==
    self.A_param @ self.x_var[:, k] + self.B_param @ self.u_var[:, k]
]
```

控制约束：

```python
constraints += [
    self.u_var[0, k] >= self.a_min,
    self.u_var[0, k] <= self.a_max,
]
```

收紧间距约束：

```python
constraints += [
    gap_bar >= self.gap_min + self.omega_s_param,
    gap_bar <= self.gap_max - self.omega_s_param,
]
```

收紧速度约束：

```python
constraints += [
    v_ego_bar >= self.v_min + self.omega_v_param,
    v_ego_bar <= self.v_max - self.omega_v_param,
]
```

终端代价：

```python
cost += cp.quad_form(self.x_var[:, self.N], self.Q)
```

构建问题：

```python
self.mpc_problem = cp.Problem(cp.Minimize(cost), constraints)
```

### 10.8 _solve_mpc()

输入：

```python
def _solve_mpc(self, x0, desired_gap, omega_s, omega_v):
```

更新参数：

```python
A, B = self._build_model()

self.x0_param.value = x0
self.A_param.value = A
self.B_param.value = B
self.u_prev_param.value = self.u_prev
self.desired_gap_param.value = desired_gap
self.front_speed_param.value = self.front_state["speed"]
self.omega_s_param.value = omega_s
self.omega_v_param.value = omega_v
```

求解：

```python
try:
    self.mpc_problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
except cp.error.SolverError:
    print("[FollowerLonMPC] 求解异常，保持上一帧")
    return self._fallback_seq(x0)
```

保底：

```python
if self.u_var.value is None or self.x_var.value is None:
    print("[FollowerLonMPC] 求解失败，保持上一帧")
    return self._fallback_seq(x0)
```

成功返回：

```python
u_bar_seq = np.array(self.u_var.value).reshape(-1)
x_bar_seq = np.array(self.x_var.value).T
return u_bar_seq, x_bar_seq
```

### 10.9 求解失败回退

第三步可以增加一个私有函数：

```python
def _fallback_seq(self, x0):
```

含义：

```text
QP 不可行或求解异常时，使用上一周期实际输出 u_prev 生成保底控制序列。
```

实现：

```python
u_safe = np.clip(self.u_prev, self.a_min, self.a_max)
u_bar_seq = np.full(self.N, u_safe)
x_bar_seq = np.tile(x0, (self.N + 1, 1))
return u_bar_seq, x_bar_seq
```

说明：

```text
这是最小保底逻辑，不做复杂安全策略。
```

### 10.10 Tube 反馈修正

MPC 求解后：

```python
u_bar_0 = u_bar_seq[0]
x_bar_0 = x_bar_seq[0]
z0 = x - x_bar_0
u_feedback = float(self.K @ z0)
```

输出：

```python
a_cmd = u_bar_0 + u_feedback
a_cmd = np.clip(a_cmd, self.a_min, self.a_max)
```

更新：

```python
self.u_prev = a_cmd
```

### 10.11 debug_info

第三步返回：

```python
debug_info = {
    "tau": tau,
    "h": h,
    "desired_gap": desired_gap,
    "gap": gap,
    "e_s": e_s,
    "e_v": e_v,
    "omega_s": omega_s,
    "omega_v": omega_v,
    "triggered": True,
    "trigger_error_norm": 0.0,
    "u_bar": u_bar_0,
    "u_feedback": u_feedback,
    "a_cmd": a_cmd,
}
```

说明：

```text
triggered 第三步固定为 True。
第四步接入事件触发后再变为真实值。
```

### 10.12 第三步验收

只做 controller 单元级验收，不启动 ROS 节点：

```text
1. platoon_controller.py 语法检查通过。
2. FollowerLonMPC 可以实例化。
3. compute() 返回 (float, dict)。
4. tau=0.10 时 desired_gap > tau=0.01。
5. tau=0.10 时 omega_s / omega_v > tau=0.01。
6. a_cmd 在 [a_min, a_max] 内。
```

注意：

```text
第三步完成后 platoon_node.py 仍未适配新返回值。
因此第三步后不要直接运行完整 ROS 控制节点。
```
