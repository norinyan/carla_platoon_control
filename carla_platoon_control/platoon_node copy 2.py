"""
三车编队控制节点。

终端1：启动 CARLA
carla

终端2：生成编队场景
ros2 launch carla_platoon_control spawn_scene.launch.py route:=A

终端3：启动三车编队控制节点
ros2 run carla_platoon_control platoon_node --ros-args -p route:=A

终端4：启动可视化
ros2 run carla_platoon_control visualizer route:=A

终端5：发送启动信号
ros2 topic pub --once /carla_platoon/start std_msgs/Bool "data: true"
"""

import os
import csv
import math
import yaml

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

        # timer
        self.timer = self.create_timer(self.ctrl_period, self._on_timer)


    def _on_timer(self):
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

        # 3. WAITING：等待启动信号，三车保持刹车
        if self.tracker_state == self.STATE_WAITING:
            self._pub_all_hold()
            self.get_logger().info(
                "等待启动信号 /carla_platoon/start ...",
                throttle_duration_sec=3.0,
            )
            return

        # 4. DRIVING：正常行驶，同时检查是否需要停车
        if self.tracker_state == self.STATE_DRIVING:
            if self._is_ready_to_stop():
                self.tracker_state = self.STATE_STOPPING
                self.get_logger().info("state: DRIVING -> STOPPING")
                self._handle_stopping()
                return

        # 5. STOPPING：三车统一刹车
        if self.tracker_state == self.STATE_STOPPING:
            self._handle_stopping()
            return

        # 6. STOPPED：停稳后保持刹车
        if self.tracker_state == self.STATE_STOPPED:
            self._pub_all_hold()
            self.get_logger().info(
                "state=STOPPED，保持刹车",
                throttle_duration_sec=2.0,
            )
            return

        # 7. 取三辆车状态
        state_0 = self.vehicle_states["vehicle_0"]
        state_1 = self.vehicle_states["vehicle_1"]
        state_2 = self.vehicle_states["vehicle_2"]

        # 8. 三辆车各自找参考点
        ref_points_0 = self._find_ref_points("vehicle_0")
        ref_points_1 = self._find_ref_points("vehicle_1")
        ref_points_2 = self._find_ref_points("vehicle_2")

        # 9. 横向控制
        steer_0 = self.lat_ctrl_0.compute(state_0, ref_points_0, self.ctrl_period)
        steer_1 = self.lat_ctrl_1.compute(state_1, ref_points_1, self.ctrl_period)
        steer_2 = self.lat_ctrl_2.compute(state_2, ref_points_2, self.ctrl_period)

        # 10. 纵向控制
        accel_0 = self.leader_lon_ctrl.compute(
            state_0,
            ref_points_0,
            self.ctrl_period,
        )

        accel_1 = self.follower_lon_ctrl_1.compute(
            state_1,
            ref_points_1,
            state_0,
            0.0,
            self.ctrl_period,
        )

        accel_2 = self.follower_lon_ctrl_2.compute(
            state_2,
            ref_points_2,
            state_1,
            0.0,
            self.ctrl_period,
        )

        # 11. a_cmd + steer 映射为 throttle / brake / steer
        ctrl_cmd_0 = self._map2cmd(accel_0, steer_0, state_0)
        ctrl_cmd_1 = self._map2cmd(accel_1, steer_1, state_1)
        ctrl_cmd_2 = self._map2cmd(accel_2, steer_2, state_2)

        # 12. 发布三车控制命令
        self._pub_cmd("vehicle_0", ctrl_cmd_0)
        self._pub_cmd("vehicle_1", ctrl_cmd_1)
        self._pub_cmd("vehicle_2", ctrl_cmd_2)

        self.get_logger().info(
            f"state=DRIVING, "
            f"v0={state_0['speed']:.2f}, "
            f"v1={state_1['speed']:.2f}, "
            f"v2={state_2['speed']:.2f}, "
            f"a0={accel_0:.2f}, "
            f"a1={accel_1:.2f}, "
            f"a2={accel_2:.2f}",
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
        node.destroy_node()
        rclpy.shutdown()
        

if __name__ == "__main__":
    main()
