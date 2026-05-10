# DTE 编队控制 MVP 修改规划

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

