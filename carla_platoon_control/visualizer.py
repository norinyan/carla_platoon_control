import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray

import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
from datetime import datetime
from pathlib import Path


VEHICLE_IDS = ["vehicle_0", "vehicle_1", "vehicle_2"]
VEHICLE_COLORS = {
    "vehicle_0": "#4fc3f7",
    "vehicle_1": "#ffb74d",
    "vehicle_2": "#ce93d8",
}


class Visualizer(Node):

    def __init__(self, ref_path_x, ref_path_y, route):
        super().__init__("platoon_visualizer")

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.subs = []
        for vehicle_id in VEHICLE_IDS:
            self.subs.append(
                self.create_subscription(
                    Float32MultiArray,
                    f"/carla_platoon/{vehicle_id}/control_state",
                    self._make_callback(vehicle_id),
                    sensor_qos,
                )
            )

        # 参考路径（给小地图用）
        self.ref_path_x = ref_path_x
        self.ref_path_y = ref_path_y
        self.route = route

        # 数据缓冲：按需求保留全部帧
        self.data = {}
        for vehicle_id in VEHICLE_IDS:
            self.data[vehicle_id] = {
                "t": [],
                "speed_now": [],
                "speed_ref": [],
                "yaw_err": [],
                "steer": [],
                "accel": [],
                "traj_x": [],
                "traj_y": [],
                "s": [],
                "gap": [],
                "desired_gap": [],
                "throttle": [],
                "brake": [],
                "frame": 0,
            }

        self.latest = {}

        # 记录到 data/*.csv
        data_dir = Path("/home/nor/ros2_carla_ws/src/carla_platoon_control/data")
        data_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = data_dir / f"platoon_visualizer_route_{self.route}_{ts}.csv"
        self.csv_fp = self.csv_path.open("w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_fp)
        self.csv_writer.writerow([
            "t",
            "vehicle_id",
            "speed_now",
            "speed_ref",
            "yaw_now",
            "yaw_ref",
            "yaw_err_deg",
            "steer",
            "accel",
            "traj_x",
            "traj_y",
            "s",
            "gap",
            "desired_gap",
            "throttle",
            "brake",
        ])
        self.csv_fp.flush()
        self.get_logger().info(f"recording csv -> {self.csv_path}")

    def _make_callback(self, vehicle_id):
        def _on_data(msg):
            print(f"收到{vehicle_id}数据: {len(msg.data)} 个字段")
            d = msg.data
            if len(d) < 13:
                return

            buf = self.data[vehicle_id]
            buf["frame"] += 1
            t = buf["frame"] * 0.1  # 10Hz

            buf["t"].append(t)
            buf["speed_now"].append(d[0])
            buf["speed_ref"].append(d[1])

            err = d[3] - d[2]
            err = math.atan2(math.sin(err), math.cos(err))
            buf["yaw_err"].append(math.degrees(err))

            buf["steer"].append(d[4])
            buf["accel"].append(d[5])
            buf["traj_x"].append(d[6])
            buf["traj_y"].append(d[7])
            buf["s"].append(d[8])
            buf["gap"].append(d[9])
            buf["desired_gap"].append(d[10])
            buf["throttle"].append(d[11])
            buf["brake"].append(d[12])

            self.latest[vehicle_id] = d

            self.csv_writer.writerow([
                t,
                vehicle_id,
                d[0],
                d[1],
                d[2],
                d[3],
                buf["yaw_err"][-1],
                d[4],
                d[5],
                d[6],
                d[7],
                d[8],
                d[9],
                d[10],
                d[11],
                d[12],
            ])
            self.csv_fp.flush()

        return _on_data

    def close_recorder(self):
        if hasattr(self, "csv_fp") and self.csv_fp and not self.csv_fp.closed:
            self.csv_fp.flush()
            self.csv_fp.close()


def build_figure():
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("#1e1e2e")

    # 布局：左列4个数据图，右列1个大地图
    ax_speed  = fig.add_subplot(4, 2, 1)
    ax_gap    = fig.add_subplot(4, 2, 3)
    ax_steer  = fig.add_subplot(4, 2, 5)
    ax_accel  = fig.add_subplot(4, 2, 7)
    ax_map    = fig.add_subplot(1, 2, 2)

    for ax in [ax_speed, ax_gap, ax_steer, ax_accel, ax_map]:
        ax.set_facecolor("#2a2a3e")
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#555577")

    ax_speed.set_title("Speed (m/s)",  color="white", fontsize=9)
    ax_gap.set_title("Net Gap (m)", color="white", fontsize=9)
    ax_steer.set_title("Steer Output", color="white", fontsize=9)
    ax_accel.set_title("Accel Output (m/s²)", color="white", fontsize=9)
    ax_map.set_title("Platoon Trajectory Map", color="white", fontsize=9)
    ax_map.set_aspect("equal")

    fig.tight_layout(pad=2.0)
    return fig, ax_speed, ax_gap, ax_steer, ax_accel, ax_map


def main(args=None):
    rclpy.init(args=args)

    # 读参考路径（给小地图画底图用）
    import os
    from ament_index_python.packages import get_package_share_directory

    # 从参数或环境变量读 route，默认 A
    import sys
    route = "A"
    for arg in sys.argv:
        if arg.startswith("route:="):
            route = arg.split(":=")[1].upper()

    pkg_share = get_package_share_directory("carla_platoon_control")
    csv_path = os.path.join(pkg_share, "config", f"route_{route}.csv")
    ref_x, ref_y = [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref_x.append(float(row["x"]))
            ref_y.append(float(row["y"]))

    node = Visualizer(ref_x, ref_y, route)

    fig, ax_speed, ax_gap, ax_steer, ax_accel, ax_map = build_figure()

    def update(_):
        rclpy.spin_once(node, timeout_sec=0)

        data = node.data
        if all(len(data[vehicle_id]["t"]) == 0 for vehicle_id in VEHICLE_IDS):
            return

        # 速度
        ax_speed.cla()
        ax_speed.set_facecolor("#2a2a3e")
        ax_speed.set_title("Speed (m/s)", color="white", fontsize=9)
        for vehicle_id in VEHICLE_IDS:
            buf = data[vehicle_id]
            if len(buf["t"]) == 0:
                continue
            color = VEHICLE_COLORS[vehicle_id]
            if vehicle_id == "vehicle_0":
                ax_speed.plot(buf["t"], buf["speed_ref"], "--", color="#888888",
                              linewidth=1, label="ref")
            ax_speed.plot(buf["t"], buf["speed_now"], color=color,
                          linewidth=1.2, label=vehicle_id)
        ax_speed.tick_params(colors="white", labelsize=7)
        ax_speed.legend(fontsize=7,
                facecolor="#1e1e2e", edgecolor="none")

        # 净间距
        ax_gap.cla()
        ax_gap.set_facecolor("#2a2a3e")
        ax_gap.set_title("Net Gap (m)", color="white", fontsize=9)
        v1 = data["vehicle_1"]
        v2 = data["vehicle_2"]
        if len(v1["t"]) > 0:
            ax_gap.plot(v1["t"], v1["gap"], color="#ffb74d",
                        linewidth=1.2, label="gap01")
            ax_gap.plot(v1["t"], v1["desired_gap"], "--", color="#888888",
                        linewidth=1, label="desired")
        if len(v2["t"]) > 0:
            ax_gap.plot(v2["t"], v2["gap"], color="#ce93d8",
                        linewidth=1.2, label="gap12")
        ax_gap.axhline(0, color="#555577", linewidth=0.8, linestyle="--")
        ax_gap.tick_params(colors="white", labelsize=7)
        ax_gap.legend(fontsize=7,
                facecolor="#1e1e2e", edgecolor="none")

        # 转向输出
        ax_steer.cla()
        ax_steer.set_facecolor("#2a2a3e")
        ax_steer.set_title("Steer Output [-1,1]", color="white", fontsize=9)
        for vehicle_id in VEHICLE_IDS:
            buf = data[vehicle_id]
            if len(buf["t"]) == 0:
                continue
            ax_steer.plot(buf["t"], buf["steer"], color=VEHICLE_COLORS[vehicle_id],
                          linewidth=1.2, label=vehicle_id)
        ax_steer.axhline(0, color="#555577", linewidth=0.8, linestyle="--")
        ax_steer.set_ylim(-1.1, 1.1)
        ax_steer.tick_params(colors="white", labelsize=7)
        ax_steer.legend(fontsize=7,
                facecolor="#1e1e2e", edgecolor="none")

        # 纵向输出
        ax_accel.cla()
        ax_accel.set_facecolor("#2a2a3e")
        ax_accel.set_title("Accel Output (m/s²)", color="white", fontsize=9)
        for vehicle_id in VEHICLE_IDS:
            buf = data[vehicle_id]
            if len(buf["t"]) == 0:
                continue
            ax_accel.plot(buf["t"], buf["accel"], color=VEHICLE_COLORS[vehicle_id],
                          linewidth=1.2, label=vehicle_id)
        ax_accel.axhline(0, color="#555577", linewidth=0.8, linestyle="--")
        ax_accel.tick_params(colors="white", labelsize=7)
        ax_accel.legend(fontsize=7,
                facecolor="#1e1e2e", edgecolor="none")

        # 小地图
        ax_map.cla()
        ax_map.set_facecolor("#2a2a3e")
        ax_map.set_title("Platoon Trajectory Map", color="white", fontsize=9)
        ax_map.set_aspect("equal")
        ax_map.plot(ref_x, ref_y, "--", color="#ff9800",
            linewidth=1, label="ref path")
        for vehicle_id in VEHICLE_IDS:
            buf = data[vehicle_id]
            color = VEHICLE_COLORS[vehicle_id]
            if len(buf["traj_x"]) > 1:
                ax_map.plot(buf["traj_x"], buf["traj_y"], color=color,
                            linewidth=1.2, label=vehicle_id)
            if len(buf["traj_x"]) > 0:
                ax_map.plot(buf["traj_x"][-1], buf["traj_y"][-1], "o",
                            color=color, markersize=6)
        ax_map.tick_params(colors="white", labelsize=7)
        ax_map.legend(fontsize=7,
                    facecolor="#1e1e2e", edgecolor="none")
        fig.canvas.draw_idle()

    ani = animation.FuncAnimation(fig, update, interval=100, cache_frame_data=False)
    node.ani = ani
    plt.show()

    node.close_recorder()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
