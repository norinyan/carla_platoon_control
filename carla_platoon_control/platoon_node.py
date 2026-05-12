"""
三车编队控制节点。

终端1：启动 CARLA
carla

终端2：生成编队场景
ros2 launch carla_platoon_control spawn_scene.launch.py route:=A

终端3：启动三车编队控制节点（会在 src/carla_platoon_control/data/ 下记录实验 CSV）
ros2 run carla_platoon_control platoon_node --ros-args -p route:=A

终端4：启动实时可视化（订阅 /carla_platoon/experiment_state，不再负责记录 CSV）
ros2 run carla_platoon_control visualizer --ros-args --remap route:=A

终端5：发送启动信号
ros2 topic pub --once /carla_platoon/start std_msgs/Bool "data: true"

离线复盘：读取 platoon_node 记录的 CSV 并生成论文指标图
ros2 run carla_platoon_control visualizer csv:=/home/nor/ros2_carla_ws/src/carla_platoon_control/data/platoon_control_route_A_YYYYMMDD_HHMMSS.csv --ros-args --remap route:=A
"""

import os
import csv
import json
import math
import random
import yaml
from datetime import datetime
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from carla_msgs.msg import CarlaEgoVehicleControl
from nav_msgs.msg import Odometry

from carla_platoon_control.controllers.platoon_controller import LatMPC
from carla_platoon_control.controllers.platoon_controller import LeaderLonPID
from carla_platoon_control.controllers.platoon_controller import FollowerLonMPC

from std_msgs.msg import Bool
from std_msgs.msg import String

class PlatoonControlNode(Node):
    def __init__(self):
        super().__init__("platoon_control_node")

        # load config yaml 
        pkg_share = get_package_share_directory("carla_platoon_control")
        params_path = os.path.join(pkg_share, "config", "controller_params.yaml")

        with open(params_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)


        # init Qos
        sensor_qos = QoSProfile(
            reliability = ReliabilityPolicy.BEST_EFFORT, 
            history = HistoryPolicy.KEEP_LAST, 
            depth = 10, 
        )

        # publisher
        self.pub_cmd_0 = self.create_publisher(
            CarlaEgoVehicleControl,
            "/carla/vehicle_0/vehicle_control_cmd",
            10,
        )

        self.pub_cmd_1 = self.create_publisher(
            CarlaEgoVehicleControl,
            "/carla/vehicle_1/vehicle_control_cmd",
            10,
        )

        self.pub_cmd_2 = self.create_publisher(
            CarlaEgoVehicleControl,
            "/carla/vehicle_2/vehicle_control_cmd",
            10,
        )

        self.pub_experiment_state = self.create_publisher(
            String,
            "/carla_platoon/experiment_state",
            10,
        )

        # subscriber
        self.sub_odom_0 = self.create_subscription(
            Odometry,
            "/carla/vehicle_0/odometry_vehicle_0",
            lambda msg: self._on_odom(msg, "vehicle_0"),
            sensor_qos,
        )

        self.sub_odom_1 = self.create_subscription(
            Odometry,
            "/carla/vehicle_1/odometry_vehicle_1",
            lambda msg: self._on_odom(msg, "vehicle_1"),
            sensor_qos,
        )

        self.sub_odom_2 = self.create_subscription(
            Odometry,
            "/carla/vehicle_2/odometry_vehicle_2",
            lambda msg: self._on_odom(msg, "vehicle_2"),
            sensor_qos,
        )

        self.sub_start = self.create_subscription(
            Bool,
            "/carla_platoon/start",
            self._on_start,
            10,
        )

        # FSM 状态机
        self.STATE_WAITING = "WAITING"
        self.STATE_DRIVING = "DRIVING"
        self.STATE_STOPPING = "STOPPING"
        self.STATE_STOPPED = "STOPPED"

        self.tracker_state = self.STATE_WAITING

        # 停车参数
        self.stop_distance = 2.0
        self.stop_speed = 0.2
        self.stop_brake_hold = 0.3

        # 车辆状态
        self.vehicle_states = {
            "vehicle_0": {
                "x": 0.0,
                "y": 0.0,
                "yaw": 0.0,
                "speed": 0.0,
                "s": 0.0,
            },
            "vehicle_1": {
                "x": 0.0,
                "y": 0.0,
                "yaw": 0.0,
                "speed": 0.0,
                "s": 0.0,
            },
            "vehicle_2": {
                "x": 0.0,
                "y": 0.0,
                "yaw": 0.0,
                "speed": 0.0,
                "s": 0.0,
            },
        }

        # odom 是否已经收到
        self.odom_ready = {
            "vehicle_0": False,
            "vehicle_1": False,
            "vehicle_2": False,
        }

        self.ref_traj = []
        self.idx_ref = {
                "vehicle_0": 0,
                "vehicle_1": 0,
                "vehicle_2": 0,
            }

        self.ref_points_num = 30
        
        self.declare_parameter("route", "A")
        self.route = str(self.get_parameter("route").value).upper()
        self._load_ref_traj()
        self._load_calibration_table()

        self.ctrl_period = 0.1
        self.lat_ctrl_0 = LatMPC(cfg["lat_mpc"])
        self.lat_ctrl_1 = LatMPC(cfg["lat_mpc"])
        self.lat_ctrl_2 = LatMPC(cfg["lat_mpc"])

        self.leader_lon_ctrl = LeaderLonPID(cfg["leader_lon_pid"])

        self.follower_lon_ctrl_1 = FollowerLonMPC(cfg["follower_lon_mpc"])
        self.follower_lon_ctrl_2 = FollowerLonMPC(cfg["follower_lon_mpc"])

        self.BRAKE_K = 0.2
        self._init_experiment_recorder()

        # timer
        self.timer = self.create_timer(self.ctrl_period, self._on_timer)

    def _on_timer(self):
        info_1 = None
        info_2 = None

        # 1. 参考轨迹检查
        if len(self.ref_traj) == 0:
            return

        # 2. 等三辆车 odom 都收到
        if (
            not self.odom_ready["vehicle_0"]
            or not self.odom_ready["vehicle_1"]
            or not self.odom_ready["vehicle_2"]
        ):
            self.get_logger().warn(
                "waiting odom for all vehicles...",
                throttle_duration_sec=2.0,
            )
            return

        # 3. 取三辆车状态
        state_0 = self.vehicle_states["vehicle_0"]
        state_1 = self.vehicle_states["vehicle_1"]
        state_2 = self.vehicle_states["vehicle_2"]

        ref_points_0 = self._find_ref_points("vehicle_0")
        ref_points_1 = self._find_ref_points("vehicle_1")
        ref_points_2 = self._find_ref_points("vehicle_2")

        accel_0 = None
        accel_1 = None
        accel_2 = None

        # 4. WAITING：等待启动，持续发布刹车
        if self.tracker_state == self.STATE_WAITING:
            ctrl_cmd_0 = {"throttle": 0.0, "brake": 0.3, "steer": 0.0}
            ctrl_cmd_1 = {"throttle": 0.0, "brake": 0.3, "steer": 0.0}
            ctrl_cmd_2 = {"throttle": 0.0, "brake": 0.3, "steer": 0.0}

        # 5. DRIVING：正常控制
        elif self.tracker_state == self.STATE_DRIVING:
            if self._is_ready_to_stop():
                self.tracker_state = self.STATE_STOPPING
                self.get_logger().info("state: DRIVING -> STOPPING")

                ctrl_cmd_0 = {"throttle": 0.0, "brake": 0.3, "steer": 0.0}
                ctrl_cmd_1 = {"throttle": 0.0, "brake": 0.3, "steer": 0.0}
                ctrl_cmd_2 = {"throttle": 0.0, "brake": 0.3, "steer": 0.0}

            else:
                steer_0 = self.lat_ctrl_0.compute(state_0, ref_points_0, self.ctrl_period)
                steer_1 = self.lat_ctrl_1.compute(state_1, ref_points_1, self.ctrl_period)
                steer_2 = self.lat_ctrl_2.compute(state_2, ref_points_2, self.ctrl_period)

                accel_0 = self.leader_lon_ctrl.compute(
                    state_0,
                    ref_points_0,
                    self.ctrl_period,
                )

                tau_01 = self._get_v2x_delay("vehicle_0", "vehicle_1")
                tau_12 = self._get_v2x_delay("vehicle_1", "vehicle_2")

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

                ctrl_cmd_0 = self._map2cmd(accel_0, steer_0, state_0)
                ctrl_cmd_1 = self._map2cmd(accel_1, steer_1, state_1)
                ctrl_cmd_2 = self._map2cmd(accel_2, steer_2, state_2)

        # 6. STOPPING：持续发布停车命令
        elif self.tracker_state == self.STATE_STOPPING:
            ctrl_cmd_0 = {"throttle": 0.0, "brake": 0.3, "steer": 0.0}
            ctrl_cmd_1 = {"throttle": 0.0, "brake": 0.3, "steer": 0.0}
            ctrl_cmd_2 = {"throttle": 0.0, "brake": 0.3, "steer": 0.0}

            max_speed = max(
                state_0["speed"],
                state_1["speed"],
                state_2["speed"],
            )

            if max_speed < self.stop_speed:
                self.tracker_state = self.STATE_STOPPED
                self.get_logger().info("state: STOPPING -> STOPPED")

        # 7. STOPPED：持续保持刹车
        elif self.tracker_state == self.STATE_STOPPED:
            ctrl_cmd_0 = {"throttle": 0.0, "brake": 0.3, "steer": 0.0}
            ctrl_cmd_1 = {"throttle": 0.0, "brake": 0.3, "steer": 0.0}
            ctrl_cmd_2 = {"throttle": 0.0, "brake": 0.3, "steer": 0.0}

        # 8. 未知状态：安全停车
        else:
            ctrl_cmd_0 = {"throttle": 0.0, "brake": 0.3, "steer": 0.0}
            ctrl_cmd_1 = {"throttle": 0.0, "brake": 0.3, "steer": 0.0}
            ctrl_cmd_2 = {"throttle": 0.0, "brake": 0.3, "steer": 0.0}

        # 9. 不管什么状态，最后都发布
        self._pub_cmd("vehicle_0", ctrl_cmd_0)
        self._pub_cmd("vehicle_1", ctrl_cmd_1)
        self._pub_cmd("vehicle_2", ctrl_cmd_2)

        row = self._build_experiment_row(
            state_0,
            state_1,
            state_2,
            ref_points_0,
            info_1,
            info_2,
            accel_0,
            accel_1,
            accel_2,
            ctrl_cmd_0,
            ctrl_cmd_1,
            ctrl_cmd_2,
        )
        self._write_experiment_row(row)
        self._publish_experiment_state(row)

        self.get_logger().info(
            f"state={self.tracker_state}, "
            f"v0={state_0['speed']:.2f}, "
            f"v1={state_1['speed']:.2f}, "
            f"v2={state_2['speed']:.2f}, "
            f"cmd0=({ctrl_cmd_0['throttle']:.2f}, {ctrl_cmd_0['brake']:.2f}, {ctrl_cmd_0['steer']:.2f}), "
            f"cmd1=({ctrl_cmd_1['throttle']:.2f}, {ctrl_cmd_1['brake']:.2f}, {ctrl_cmd_1['steer']:.2f}), "
            f"cmd2=({ctrl_cmd_2['throttle']:.2f}, {ctrl_cmd_2['brake']:.2f}, {ctrl_cmd_2['steer']:.2f}), "
            f"f1={self._format_follower_info(info_1)}, "
            f"f2={self._format_follower_info(info_2)}",
            throttle_duration_sec=1.0,
        )

    def _on_odom(self, msg, vehicle_id):
        state = self.vehicle_states[vehicle_id]

        # position
        state["x"] = msg.pose.pose.position.x
        state["y"] = msg.pose.pose.position.y

        # yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        state["yaw"] = math.atan2(siny_cosp, cosy_cosp)

        # speed
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
        state["speed"] = math.sqrt(vx * vx + vy * vy + vz * vz)

        # odom ready
        self.odom_ready[vehicle_id] = True

    def _on_start(self, msg):
        """收到启动信号后，从 WAITING 进入 DRIVING。"""
        if msg.data and self.tracker_state == self.STATE_WAITING:
            self.tracker_state = self.STATE_DRIVING
            self.leader_lon_ctrl.reset()
            self.get_logger().info("收到启动信号，三车开始控制")

    def _get_v2x_delay(self, front_id, rear_id):
        return random.uniform(0.01, 0.10)

    def _format_follower_info(self, info):
        if info is None:
            return "none"

        return (
            f"tau={info['tau']:.3f}, "
            f"gap={info['gap']:.2f}, "
            f"des={info['desired_gap']:.2f}, "
            f"e_s={info['e_s']:.2f}, "
            f"e_v={info['e_v']:.2f}, "
            f"trig={info['triggered']}, "
            f"a={info['a_cmd']:.2f}, "
            f"solver={info.get('solver_status', 'na')}"
        )

    def _init_experiment_recorder(self):
        data_dir = Path("/home/nor/ros2_carla_ws/src/carla_platoon_control/data")
        data_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = data_dir / f"platoon_control_route_{self.route}_{ts}.csv"
        self.csv_fp = self.csv_path.open("w", newline="", encoding="utf-8")

        self.experiment_fields = [
            "t", "state", "route",
            "x0", "y0", "yaw0", "v0", "s0",
            "x1", "y1", "yaw1", "v1", "s1",
            "x2", "y2", "yaw2", "v2", "s2",
            "speed_ref0",
            "gap01", "des01", "es01", "ev01", "tau01", "h1",
            "omega_s1", "omega_v1", "triggered1",
            "tube_triggered1", "delay_triggered1", "trigger_norm1",
            "u_bar1", "u_feedback1", "a1", "solver_status1",
            "solver_value1", "solve_time_ms1", "fallback_used1",
            "gap12", "des12", "es12", "ev12", "tau12", "h2",
            "omega_s2", "omega_v2", "triggered2",
            "tube_triggered2", "delay_triggered2", "trigger_norm2",
            "u_bar2", "u_feedback2", "a2", "solver_status2",
            "solver_value2", "solve_time_ms2", "fallback_used2",
            "a0",
            "cmd0_throttle", "cmd0_brake", "cmd0_steer",
            "cmd1_throttle", "cmd1_brake", "cmd1_steer",
            "cmd2_throttle", "cmd2_brake", "cmd2_steer",
        ]

        self.csv_writer = csv.DictWriter(self.csv_fp, fieldnames=self.experiment_fields)
        self.csv_writer.writeheader()
        self.csv_fp.flush()
        self.start_time = self.get_clock().now()
        self.get_logger().info(f"recording control csv -> {self.csv_path}")

    def _info_value(self, info, key, default=""):
        if info is None:
            return default
        value = info.get(key, default)
        if isinstance(value, (bool, str)):
            return value
        if value is None:
            return default
        return float(value)

    def _build_experiment_row(
        self,
        state_0,
        state_1,
        state_2,
        ref_points_0,
        info_1,
        info_2,
        accel_0,
        accel_1,
        accel_2,
        ctrl_cmd_0,
        ctrl_cmd_1,
        ctrl_cmd_2,
    ):
        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds * 1e-9
        speed_ref0 = ref_points_0[0]["speed"] if len(ref_points_0) > 0 else ""

        return {
            "t": t,
            "state": self.tracker_state,
            "route": self.route,
            "x0": state_0["x"], "y0": state_0["y"], "yaw0": state_0["yaw"],
            "v0": state_0["speed"], "s0": state_0["s"],
            "x1": state_1["x"], "y1": state_1["y"], "yaw1": state_1["yaw"],
            "v1": state_1["speed"], "s1": state_1["s"],
            "x2": state_2["x"], "y2": state_2["y"], "yaw2": state_2["yaw"],
            "v2": state_2["speed"], "s2": state_2["s"],
            "speed_ref0": speed_ref0,
            "gap01": self._info_value(info_1, "gap"),
            "des01": self._info_value(info_1, "desired_gap"),
            "es01": self._info_value(info_1, "e_s"),
            "ev01": self._info_value(info_1, "e_v"),
            "tau01": self._info_value(info_1, "tau"),
            "h1": self._info_value(info_1, "h"),
            "omega_s1": self._info_value(info_1, "omega_s"),
            "omega_v1": self._info_value(info_1, "omega_v"),
            "triggered1": self._info_value(info_1, "triggered"),
            "tube_triggered1": self._info_value(info_1, "tube_triggered"),
            "delay_triggered1": self._info_value(info_1, "delay_triggered"),
            "trigger_norm1": self._info_value(info_1, "trigger_error_norm"),
            "u_bar1": self._info_value(info_1, "u_bar"),
            "u_feedback1": self._info_value(info_1, "u_feedback"),
            "a1": self._info_value(info_1, "a_cmd", "" if accel_1 is None else accel_1),
            "solver_status1": self._info_value(info_1, "solver_status"),
            "solver_value1": self._info_value(info_1, "solver_value"),
            "solve_time_ms1": self._info_value(info_1, "solve_time_ms"),
            "fallback_used1": self._info_value(info_1, "fallback_used"),
            "gap12": self._info_value(info_2, "gap"),
            "des12": self._info_value(info_2, "desired_gap"),
            "es12": self._info_value(info_2, "e_s"),
            "ev12": self._info_value(info_2, "e_v"),
            "tau12": self._info_value(info_2, "tau"),
            "h2": self._info_value(info_2, "h"),
            "omega_s2": self._info_value(info_2, "omega_s"),
            "omega_v2": self._info_value(info_2, "omega_v"),
            "triggered2": self._info_value(info_2, "triggered"),
            "tube_triggered2": self._info_value(info_2, "tube_triggered"),
            "delay_triggered2": self._info_value(info_2, "delay_triggered"),
            "trigger_norm2": self._info_value(info_2, "trigger_error_norm"),
            "u_bar2": self._info_value(info_2, "u_bar"),
            "u_feedback2": self._info_value(info_2, "u_feedback"),
            "a2": self._info_value(info_2, "a_cmd", "" if accel_2 is None else accel_2),
            "solver_status2": self._info_value(info_2, "solver_status"),
            "solver_value2": self._info_value(info_2, "solver_value"),
            "solve_time_ms2": self._info_value(info_2, "solve_time_ms"),
            "fallback_used2": self._info_value(info_2, "fallback_used"),
            "a0": "" if accel_0 is None else accel_0,
            "cmd0_throttle": ctrl_cmd_0["throttle"],
            "cmd0_brake": ctrl_cmd_0["brake"],
            "cmd0_steer": ctrl_cmd_0["steer"],
            "cmd1_throttle": ctrl_cmd_1["throttle"],
            "cmd1_brake": ctrl_cmd_1["brake"],
            "cmd1_steer": ctrl_cmd_1["steer"],
            "cmd2_throttle": ctrl_cmd_2["throttle"],
            "cmd2_brake": ctrl_cmd_2["brake"],
            "cmd2_steer": ctrl_cmd_2["steer"],
        }

    def _write_experiment_row(self, row):
        self.csv_writer.writerow(row)
        self.csv_fp.flush()

    def _publish_experiment_state(self, row):
        msg = String()
        msg.data = json.dumps(row, ensure_ascii=True)
        self.pub_experiment_state.publish(msg)

    def close_experiment_recorder(self):
        if hasattr(self, "csv_fp") and self.csv_fp and not self.csv_fp.closed:
            self.csv_fp.flush()
            self.csv_fp.close()

    def _dist_to_goal(self, vehicle_id):
        """计算某辆车到终点的直线距离。"""
        if len(self.ref_traj) == 0:
            return float("inf")

        vehicle_state = self.vehicle_states[vehicle_id]
        goal = self.ref_traj[-1]

        dx = goal["x"] - vehicle_state["x"]
        dy = goal["y"] - vehicle_state["y"]

        return math.hypot(dx, dy)

    def _is_ready_to_stop(self):
        """只用 leader 判断是否触发停车。"""
        return self._dist_to_goal("vehicle_0") < self.stop_distance

    def _pub_all_hold(self):
        """三辆车统一保持刹车。"""
        hold_cmd = {
            "throttle": 0.0,
            "brake": self.stop_brake_hold,
            "steer": 0.0,
        }

        self._pub_cmd("vehicle_0", hold_cmd)
        self._pub_cmd("vehicle_1", hold_cmd)
        self._pub_cmd("vehicle_2", hold_cmd)


    def _handle_stopping(self):
        """停车阶段：三辆车一起刹车，全部低速后进入 STOPPED。"""
        speed_0 = self.vehicle_states["vehicle_0"]["speed"]
        speed_1 = self.vehicle_states["vehicle_1"]["speed"]
        speed_2 = self.vehicle_states["vehicle_2"]["speed"]

        max_speed = max(speed_0, speed_1, speed_2)

        if max_speed < self.stop_speed:
            self.tracker_state = self.STATE_STOPPED
            self._pub_all_hold()
            self.get_logger().info("state: STOPPING -> STOPPED")
            return

        self._pub_all_hold()

        self.get_logger().info(
            f"state=STOPPING, max_v={max_speed:.2f}, "
            f"d_goal={self._dist_to_goal('vehicle_0'):.2f}, "
            f"brake={self.stop_brake_hold:.2f}",
            throttle_duration_sec=1.0,
        )


    def _load_ref_traj(self):
        pkg_share = get_package_share_directory("carla_platoon_control")
        csv_path = os.path.join(pkg_share, "config", f"route_{self.route}.csv")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"route file not found: {csv_path}")
        
        self.ref_traj = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.ref_traj.append({
                    "s": float(row["s"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "yaw": float(row["yaw"]),
                    "kappa": float(row["kappa"]),
                    "speed": float(row["speed"]),
                })

    def _find_ref_points(self, vehicle_id):
        vehicle_state = self.vehicle_states[vehicle_id]

        search_window = 30
        start = self.idx_ref[vehicle_id]
        end = min(start + search_window, len(self.ref_traj))

        min_dist = float("inf")
        nearest_idx = self.idx_ref[vehicle_id]

        for i in range(start, end):
            dx = self.ref_traj[i]["x"] - vehicle_state["x"]
            dy = self.ref_traj[i]["y"] - vehicle_state["y"]
            dist = math.hypot(dx, dy)

            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        self.idx_ref[vehicle_id] = nearest_idx
        vehicle_state["s"] = self.ref_traj[nearest_idx]["s"]

        end_idx = min(nearest_idx + self.ref_points_num, len(self.ref_traj))

        return self.ref_traj[nearest_idx:end_idx]

    def _load_calibration_table(self):
        pkg_share = get_package_share_directory("carla_platoon_control")
        csv_path = os.path.join(pkg_share, "config", "calibration_table.csv")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        self.calib_speed_bins = [float(x) for x in rows[0][1:]]
        self.calib_throttle_list = []
        self.calib_grid = []  # grid[i][j] = throttle_i 在 speed_bin_j 下的加速度，NaN->None
        for r in rows[1:]:
            self.calib_throttle_list.append(float(r[0]))
            vals = []
            for x in r[1:]:
                vals.append(None if x == "NaN" else float(x))
            self.calib_grid.append(vals)

        self.get_logger().info(
            f"calibration table loaded: {len(self.calib_throttle_list)} throttle levels, "
            f"{len(self.calib_speed_bins)} speed bins"
        )

    def _map2cmd(self, accel_out, steer_out, vehicle_state):

        dead_zone = 0.05
        v_now = vehicle_state["speed"]

        if accel_out >= 0.0:
            throttle_cmd = self._interp_2d(accel_out, v_now)
            throttle_cmd += 0.0153 * v_now
            brake_cmd = 0.0
        else:
            if abs(accel_out) < dead_zone:
                throttle_cmd = 0.0
                brake_cmd = 0.0
            else:
                throttle_cmd = 0.0
                brake_cmd = max(0.0, min(1.0, abs(accel_out) * self.BRAKE_K))

        steer_cmd = max(-1.0, min(1.0, steer_out))
        throttle_cmd = max(0.0, min(1.0, throttle_cmd))
        brake_cmd = max(0.0, min(1.0, brake_cmd))

        return {
            "throttle": throttle_cmd,
            "brake": brake_cmd,
            "steer": steer_cmd,
        }

    def _interp_2d(self, a_des, v_now):
        """二维查表：已知期望加速度和当前速度，插值得到油门开度。"""

        # step1: 每个throttle行，在v_now处插值得到对应加速度
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

        # step2: 在 (a_at_v, throttle) 曲线上反查 a_des 对应的 throttle
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
    
    def _pub_cmd(self, vehicle_id, ctrl_cmd):
        msg = CarlaEgoVehicleControl()

        msg.throttle = float(max(0.0, min(1.0, ctrl_cmd["throttle"])))
        msg.brake = float(max(0.0, min(1.0, ctrl_cmd["brake"])))
        msg.steer = float(max(-1.0, min(1.0, ctrl_cmd["steer"])))

        msg.hand_brake = False
        msg.reverse = False
        msg.manual_gear_shift = False
        msg.gear = 0

        if vehicle_id == "vehicle_0":
            self.pub_cmd_0.publish(msg)
        elif vehicle_id == "vehicle_1":
            self.pub_cmd_1.publish(msg)
        elif vehicle_id == "vehicle_2":
            self.pub_cmd_2.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PlatoonControlNode()
    try:
        rclpy.spin(node)
    finally:
        node.close_experiment_recorder()
        node.destroy_node()
        rclpy.shutdown()
        

if __name__ == "__main__":
    main()
