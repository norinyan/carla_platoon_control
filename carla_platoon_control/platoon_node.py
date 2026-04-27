"""
三车编队闭环控制节点。

启动场景：
ros2 launch carla_platoon_control spawn_scene.launch.py

启动控制节点：
python3 -m carla_platoon_control.platoon_node

启动可视化：
ros2 run carla_platoon_control visualizer route:=A

发送启动信号：
ros2 topic pub --once /carla_platoon/start std_msgs/Bool "data: true"


"""

import csv
import math
import os
from dataclasses import dataclass

import yaml
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from carla_msgs.msg import CarlaEgoVehicleControl
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from std_msgs.msg import Float32MultiArray

from carla_platoon_control.controllers.platoon_controller import PlatoonMPCController

try:
    import carla
except ImportError:
    carla = None


PACKAGE_NAME = "carla_platoon_control"
VEHICLE_IDS = ["vehicle_0", "vehicle_1", "vehicle_2"]


@dataclass
class VehicleRuntime:
    vehicle_id: str
    controller: PlatoonMPCController
    cmd_pub: object
    viz_pub: object
    idx_ref: int = 0
    odom_ready: bool = False
    steer_out: float = 0.0
    accel_out: float = 0.0
    throttle_cmd: float = 0.0
    brake_cmd: float = 0.0

    def __post_init__(self):
        self.state = {
            "x": 0.0,
            "y": 0.0,
            "yaw": 0.0,
            "speed": 0.0,
            "s": 0.0,
        }


class PlatoonControlNode(Node):
    def __init__(self):
        super().__init__("platoon_control_node")

        self.config_dir = self._get_config_dir()
        self.ctrl_cfg = self._load_yaml("controller_params.yaml")
        self.vehicle_cfg = self._load_yaml("vehicle_params.yaml")

        # 参数
        default_route = self.vehicle_cfg.get("route", {}).get("default", "A")
        default_dt = self.vehicle_cfg.get("scene", {}).get("fixed_delta_seconds", 0.1)
        self.declare_parameter("route", default_route)
        self.declare_parameter("ctrl_period", default_dt)
        self.declare_parameter("desired_gap", 10.0)
        self.declare_parameter("vehicle_length", 4.9)
        self.declare_parameter("ref_points_num", 30)
        self.declare_parameter("search_window", 120)
        self.declare_parameter("auto_start", False)
        self.declare_parameter("enable_spectator", True)

        self.route = str(self.get_parameter("route").value).upper()
        self.ctrl_period = float(self.get_parameter("ctrl_period").value)
        self.desired_gap = float(self.get_parameter("desired_gap").value)
        self.vehicle_length = float(self.get_parameter("vehicle_length").value)
        self.ref_points_num = int(self.get_parameter("ref_points_num").value)
        self.search_window = int(self.get_parameter("search_window").value)
        self.enable_spectator = bool(self.get_parameter("enable_spectator").value)

        self.ctrl_cfg["ctrl_period"] = self.ctrl_period
        self.ctrl_cfg["vehicle_length"] = self.vehicle_length

        self.STATE_WAITING = "WAITING"
        self.STATE_DRIVING = "DRIVING"
        self.STATE_STOPPING = "STOPPING"
        self.STATE_STOPPED = "STOPPED"
        self.tracker_state = (
            self.STATE_DRIVING
            if bool(self.get_parameter("auto_start").value)
            else self.STATE_WAITING
        )

        self.stop_distance = 2.0
        self.stop_speed = 0.2
        self.stop_brake_hold = 0.30
        self.stop_brake_gain = 0.06
        self.stop_integral = 0.0
        self.BRAKE_K = 0.2

        self.ref_traj = []
        self._load_ref_traj()
        self._load_calibration_table()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.vehicles = {}
        self.sub_odom = {}
        for vehicle_id in VEHICLE_IDS:
            cmd_topic = f"/carla/{vehicle_id}/vehicle_control_cmd"
            odom_topic = f"/carla/{vehicle_id}/odometry_{vehicle_id}"
            viz_topic = f"/carla_platoon/{vehicle_id}/control_state"

            cmd_pub = self.create_publisher(CarlaEgoVehicleControl, cmd_topic, 10)
            viz_pub = self.create_publisher(Float32MultiArray, viz_topic, 10)
            controller = PlatoonMPCController(self.ctrl_cfg)
            self._attach_calibration(controller)

            self.vehicles[vehicle_id] = VehicleRuntime(
                vehicle_id=vehicle_id,
                controller=controller,
                cmd_pub=cmd_pub,
                viz_pub=viz_pub,
            )
            self.sub_odom[vehicle_id] = self.create_subscription(
                Odometry,
                odom_topic,
                self._make_odom_callback(vehicle_id),
                sensor_qos,
            )

            self.get_logger().info(
                f"{vehicle_id}: sub {odom_topic}, pub {cmd_topic}, viz {viz_topic}"
            )

        self.sub_start = self.create_subscription(
            Bool,
            "/carla_platoon/start",
            self._on_start,
            10,
        )

        self.spectator = None
        self._init_spectator()

        self.timer = self.create_timer(self.ctrl_period, self._on_timer)

        self.get_logger().info(
            f"route={self.route}, points={len(self.ref_traj)}, "
            f"dt={self.ctrl_period:.2f}, desired_gap={self.desired_gap:.2f}, "
            f"vehicle_length={self.vehicle_length:.2f}, "
            f"state={self.tracker_state}"
        )

    # ─────────────────────────────────────────────
    #  ROS callbacks
    # ─────────────────────────────────────────────
    def _on_start(self, msg):
        if msg.data and self.tracker_state == self.STATE_WAITING:
            self.tracker_state = self.STATE_DRIVING
            self.get_logger().info("收到启动信号，三车编队开始闭环控制")

    def _make_odom_callback(self, vehicle_id):
        def _on_odom(msg):
            runtime = self.vehicles[vehicle_id]
            state = runtime.state

            state["x"] = msg.pose.pose.position.x
            state["y"] = msg.pose.pose.position.y

            q = msg.pose.pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            state["yaw"] = math.atan2(siny_cosp, cosy_cosp)

            vx = msg.twist.twist.linear.x
            vy = msg.twist.twist.linear.y
            vz = msg.twist.twist.linear.z
            state["speed"] = math.sqrt(vx * vx + vy * vy + vz * vz)

            runtime.odom_ready = True

        return _on_odom

    def _on_timer(self):
        if len(self.ref_traj) == 0:
            return

        if not self._all_odom_ready():
            self.get_logger().warn("waiting odom for all vehicles...", throttle_duration_sec=2.0)
            return

        if self.tracker_state == self.STATE_WAITING:
            self._pub_all_hold(brake=self.stop_brake_hold)
            self.get_logger().info("等待启动信号 /carla_platoon/start ...", throttle_duration_sec=3.0)
            return

        if self.tracker_state == self.STATE_DRIVING and self._is_ready_to_stop():
            self.tracker_state = self.STATE_STOPPING
            self.stop_integral = 0.0
            self.get_logger().info("state: DRIVING -> STOPPING")

        if self.tracker_state == self.STATE_STOPPING:
            self._handle_stopping()
            return

        if self.tracker_state == self.STATE_STOPPED:
            self._pub_all_hold(brake=self.stop_brake_hold)
            self.get_logger().info("state=STOPPED，保持刹车", throttle_duration_sec=1.0)
            return

        self._run_closed_loop_once()
        self._update_spectator()

    # ─────────────────────────────────────────────
    #  闭环控制
    # ─────────────────────────────────────────────
    def _run_closed_loop_once(self):
        for i, vehicle_id in enumerate(VEHICLE_IDS):
            runtime = self.vehicles[vehicle_id]
            ref_points = self._find_ref_points(runtime)

            if i == 0:
                leader_state = None
            else:
                leader_id = VEHICLE_IDS[i - 1]
                leader_state = self.vehicles[leader_id].state

            steer_out, accel_out = runtime.controller.compute(
                runtime.state,
                ref_points,
                leader_state,
                self.desired_gap,
                self.ctrl_period,
            )
            ctrl_cmd = runtime.controller.map2cmd(accel_out, steer_out)

            runtime.steer_out = steer_out
            runtime.accel_out = accel_out
            runtime.throttle_cmd = ctrl_cmd["throttle"]
            runtime.brake_cmd = ctrl_cmd["brake"]

            self._pub_cmd(runtime, ctrl_cmd)
            self._pub_viz(runtime, ref_points, leader_state)

        self._log_status()

    def _handle_stopping(self):
        max_speed = max(v.state["speed"] for v in self.vehicles.values())
        if max_speed < self.stop_speed:
            self.tracker_state = self.STATE_STOPPED
            self._pub_all_hold(brake=self.stop_brake_hold)
            self.get_logger().info("state: STOPPING -> STOPPED")
            return

        self.stop_integral += max_speed * self.ctrl_period
        brake_cmd = self.stop_brake_gain * self.stop_integral
        brake_cmd = max(self.stop_brake_hold, min(0.6, brake_cmd))

        self.get_logger().info(
            f"state=STOPPING, max_v={max_speed:.2f}, "
            f"d_goal={self._dist_to_goal('vehicle_0'):.2f}, brake={brake_cmd:.2f}",
            throttle_duration_sec=1.0,
        )
        self._pub_all_hold(brake=brake_cmd)

    # ─────────────────────────────────────────────
    #  轨迹与标定
    # ─────────────────────────────────────────────
    def _get_config_dir(self):
        try:
            pkg_share = get_package_share_directory(PACKAGE_NAME)
            return os.path.join(pkg_share, "config")
        except Exception:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            return os.path.join(script_dir, "config")

    def _load_yaml(self, filename):
        path = os.path.join(self.config_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_ref_traj(self):
        csv_path = os.path.join(self.config_dir, f"route_{self.route}.csv")
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

    def _load_calibration_table(self):
        csv_path = os.path.join(self.config_dir, "calibration_table.csv")
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        self.calib_speed_bins = [float(x) for x in rows[0][1:]]
        self.calib_throttle_list = []
        self.calib_grid = []
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

    def _attach_calibration(self, controller):
        controller.calib_speed_bins = self.calib_speed_bins
        controller.calib_throttle_list = self.calib_throttle_list
        controller.calib_grid = self.calib_grid
        controller.BRAKE_K = self.BRAKE_K

    def _find_ref_points(self, runtime):
        start = runtime.idx_ref
        end = min(start + self.search_window, len(self.ref_traj))

        min_dist = float("inf")
        nearest_idx = runtime.idx_ref

        for i in range(start, end):
            dx = self.ref_traj[i]["x"] - runtime.state["x"]
            dy = self.ref_traj[i]["y"] - runtime.state["y"]
            dist = math.hypot(dx, dy)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        runtime.idx_ref = nearest_idx
        runtime.state["s"] = self.ref_traj[nearest_idx]["s"]

        end_idx = min(nearest_idx + self.ref_points_num, len(self.ref_traj))
        return self.ref_traj[nearest_idx:end_idx]

    # ─────────────────────────────────────────────
    #  发布与可视化
    # ─────────────────────────────────────────────
    def _pub_cmd(self, runtime, ctrl_cmd):
        msg = CarlaEgoVehicleControl()
        msg.throttle = float(max(0.0, min(1.0, ctrl_cmd["throttle"])))
        msg.brake = float(max(0.0, min(1.0, ctrl_cmd["brake"])))
        msg.steer = float(max(-1.0, min(1.0, ctrl_cmd["steer"])))
        msg.hand_brake = False
        msg.reverse = False
        msg.manual_gear_shift = False
        msg.gear = 0
        runtime.cmd_pub.publish(msg)

    def _pub_all_hold(self, brake=0.3):
        for runtime in self.vehicles.values():
            self._pub_cmd(runtime, {"throttle": 0.0, "brake": brake, "steer": 0.0})

    def _pub_viz(self, runtime, ref_points, leader_state):
        ref = ref_points[0]
        if leader_state is None:
            gap = 0.0
        else:
            gap = self._net_gap(leader_state, runtime.state)

        msg = Float32MultiArray()
        msg.data = [
            float(runtime.state["speed"]),
            float(ref["speed"]),
            float(runtime.state["yaw"]),
            float(ref["yaw"]),
            float(runtime.steer_out),
            float(runtime.accel_out),
            float(runtime.state["x"]),
            float(runtime.state["y"]),
            float(runtime.state["s"]),
            float(gap),
            float(self.desired_gap),
            float(runtime.throttle_cmd),
            float(runtime.brake_cmd),
        ]
        runtime.viz_pub.publish(msg)

    def _init_spectator(self):
        if not self.enable_spectator or carla is None:
            return

        try:
            client = carla.Client("localhost", 2000)
            client.set_timeout(2.0)
            self.spectator = client.get_world().get_spectator()
        except Exception as exc:
            self.get_logger().warn(f"CARLA spectator disabled: {exc}")
            self.spectator = None

    def _update_spectator(self):
        if self.spectator is None:
            return

        leader = self.vehicles["vehicle_1"].state
        self.spectator.set_transform(carla.Transform(
            carla.Location(x=leader["x"], y=-leader["y"], z=50),
            carla.Rotation(pitch=-90, yaw=0, roll=0),
        ))

    # ─────────────────────────────────────────────
    #  状态辅助
    # ─────────────────────────────────────────────
    def _all_odom_ready(self):
        return all(runtime.odom_ready for runtime in self.vehicles.values())

    def _is_ready_to_stop(self):
        return self._dist_to_goal("vehicle_0") < self.stop_distance

    def _net_gap(self, leader_state, follower_state):
        return leader_state["s"] - follower_state["s"] - self.vehicle_length

    def _dist_to_goal(self, vehicle_id):
        goal = self.ref_traj[-1]
        state = self.vehicles[vehicle_id].state
        dx = goal["x"] - state["x"]
        dy = goal["y"] - state["y"]
        return math.hypot(dx, dy)

    def _log_status(self):
        leader = self.vehicles["vehicle_0"]
        v1 = self.vehicles["vehicle_1"]
        v2 = self.vehicles["vehicle_2"]
        gap_01 = self._net_gap(leader.state, v1.state)
        gap_12 = self._net_gap(v1.state, v2.state)

        self.get_logger().info(
            f"v0={leader.state['speed']:.2f}, v1={v1.state['speed']:.2f}, "
            f"v2={v2.state['speed']:.2f}, gap01={gap_01:.2f}, gap12={gap_12:.2f}, "
            f"a0={leader.accel_out:.2f}, a1={v1.accel_out:.2f}, a2={v2.accel_out:.2f}, "
            f"th0={leader.throttle_cmd:.2f}, br0={leader.brake_cmd:.2f}, "
            f"th1={v1.throttle_cmd:.2f}, br1={v1.brake_cmd:.2f}, "
            f"th2={v2.throttle_cmd:.2f}, br2={v2.brake_cmd:.2f}, "
            f"st0={leader.steer_out:.2f}",
            throttle_duration_sec=1.0,
        )


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
