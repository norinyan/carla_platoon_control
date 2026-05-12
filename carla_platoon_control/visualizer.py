import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String

import json
import math
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
from ament_index_python.packages import get_package_share_directory


VEHICLE_IDS = ["vehicle_0", "vehicle_1", "vehicle_2"]
VEHICLE_COLORS = {
    "vehicle_0": "#4fc3f7",
    "vehicle_1": "#ffb74d",
    "vehicle_2": "#ce93d8",
}
PAIR_COLORS = {
    "01": "#ffb74d",
    "12": "#ce93d8",
}
D_MIN = 5.0

PLOT_FIELDS = [
    "t", "state",
    "x0", "y0", "v0", "s0",
    "x1", "y1", "v1", "s1",
    "x2", "y2", "v2", "s2",
    "speed_ref0",
    "gap01", "des01", "es01", "ev01", "tau01", "h1",
    "omega_s1", "omega_v1", "triggered1", "trigger_norm1",
    "u_bar1", "u_feedback1", "a1", "solve_time_ms1", "fallback_used1",
    "gap12", "des12", "es12", "ev12", "tau12", "h2",
    "omega_s2", "omega_v2", "triggered2", "trigger_norm2",
    "u_bar2", "u_feedback2", "a2", "solve_time_ms2", "fallback_used2",
    "a0",
    "cmd0_throttle", "cmd0_brake", "cmd0_steer",
    "cmd1_throttle", "cmd1_brake", "cmd1_steer",
    "cmd2_throttle", "cmd2_brake", "cmd2_steer",
]


def _empty_plot_data():
    return {field: [] for field in PLOT_FIELDS}


def _to_float(value):
    if value in ("", None):
        return math.nan
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _to_trigger(value):
    if value in ("", None):
        return math.nan
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        return 1.0 if value.lower() == "true" else 0.0
    return 1.0 if bool(value) else 0.0


def _append_row(data, row):
    for field in PLOT_FIELDS:
        value = row.get(field, "")
        if field == "state":
            data[field].append(str(value))
        elif field.startswith("triggered") or field.startswith("fallback_used"):
            data[field].append(_to_trigger(value))
        else:
            data[field].append(_to_float(value))


def _finite(values):
    return [v for v in values if not math.isnan(v)]


def _rmse(values):
    vals = _finite(values)
    if not vals:
        return math.nan
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def _mean(values):
    vals = _finite(values)
    if not vals:
        return math.nan
    return sum(vals) / len(vals)


def _minimum(values):
    vals = _finite(values)
    if not vals:
        return math.nan
    return min(vals)


def _count(values, predicate):
    return sum(1 for value in _finite(values) if predicate(value))


class Visualizer(Node):

    def __init__(self, ref_path_x, ref_path_y, route):
        super().__init__("platoon_visualizer")

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.sub = self.create_subscription(
            String,
            "/carla_platoon/experiment_state",
            self._on_experiment_state,
            sensor_qos,
        )

        # 参考路径（给小地图用）
        self.ref_path_x = ref_path_x
        self.ref_path_y = ref_path_y
        self.route = route

        self.data = _empty_plot_data()
        self.get_logger().info("visualizing /carla_platoon/experiment_state")

    def _on_experiment_state(self, msg):
        try:
            row = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"invalid experiment_state JSON: {exc}")
            return
        _append_row(self.data, row)


def build_figure():
    fig, axes = plt.subplots(4, 2, figsize=(15, 10))
    fig.patch.set_facecolor("#1e1e2e")
    axes = axes.flatten()

    for ax in axes:
        ax.set_facecolor("#2a2a3e")
        ax.tick_params(colors="white", labelsize=7)
        ax.grid(color="#555577", alpha=0.25, linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#555577")

    fig.tight_layout(pad=2.0)
    return fig, axes


def _legend(ax):
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(fontsize=7, facecolor="#1e1e2e", edgecolor="none")
        for text in legend.get_texts():
            text.set_color("white")


def _plot(ax, x, y, label, color, linestyle="-", linewidth=1.2):
    if len(x) > 0 and len(y) > 0:
        ax.plot(x, y, linestyle=linestyle, color=color, linewidth=linewidth, label=label)


def summarize(data):
    es1_rmse = _rmse(data["es01"])
    es2_rmse = _rmse(data["es12"])
    ev1_rmse = _rmse(data["ev01"])
    ev2_rmse = _rmse(data["ev12"])
    gap_min = min(_minimum(data["gap01"]), _minimum(data["gap12"]))
    trig1 = _mean(data["triggered1"]) * 100.0
    trig2 = _mean(data["triggered2"]) * 100.0
    solve_values = data["solve_time_ms1"] + data["solve_time_ms2"]
    solve_mean = _mean(solve_values)
    solve_max = max(_finite(solve_values) or [math.nan])
    sat_count = (
        _count(data["a1"], lambda x: x >= 1.49 or x <= -1.99)
        + _count(data["a2"], lambda x: x >= 1.49 or x <= -1.99)
    )
    string_ratio_es = es2_rmse / es1_rmse if es1_rmse and not math.isnan(es1_rmse) else math.nan
    violations = _count(data["gap01"], lambda x: x < D_MIN) + _count(data["gap12"], lambda x: x < D_MIN)
    return {
        "es1_rmse": es1_rmse,
        "es2_rmse": es2_rmse,
        "ev1_rmse": ev1_rmse,
        "ev2_rmse": ev2_rmse,
        "gap_min": gap_min,
        "violations": violations,
        "trig1": trig1,
        "trig2": trig2,
        "solve_mean": solve_mean,
        "solve_max": solve_max,
        "sat_count": sat_count,
        "string_ratio_es": string_ratio_es,
    }


def format_summary(summary):
    return (
        f"e_s RMSE: f1={summary['es1_rmse']:.2f} m, f2={summary['es2_rmse']:.2f} m | "
        f"e_v RMSE: f1={summary['ev1_rmse']:.2f} m/s, f2={summary['ev2_rmse']:.2f} m/s | "
        f"min gap={summary['gap_min']:.2f} m | "
        f"violations={summary['violations']} | "
        f"trigger={summary['trig1']:.1f}%/{summary['trig2']:.1f}% | "
        f"solve={summary['solve_mean']:.2f}/{summary['solve_max']:.2f} ms | "
        f"sat={summary['sat_count']} | "
        f"f2/f1 e_s={summary['string_ratio_es']:.2f}"
    )


def draw_dashboard(fig, axes, data, ref_x, ref_y, route, offline=False):
    for ax in axes:
        ax.cla()
        ax.set_facecolor("#2a2a3e")
        ax.tick_params(colors="white", labelsize=7)
        ax.grid(color="#555577", alpha=0.25, linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#555577")

    t = data["t"]

    ax = axes[0]
    ax.set_title("Speed Tracking", color="white", fontsize=9)
    _plot(ax, t, data["speed_ref0"], "ref", "#888888", "--", 1.0)
    _plot(ax, t, data["v0"], "vehicle_0", VEHICLE_COLORS["vehicle_0"])
    _plot(ax, t, data["v1"], "vehicle_1", VEHICLE_COLORS["vehicle_1"])
    _plot(ax, t, data["v2"], "vehicle_2", VEHICLE_COLORS["vehicle_2"])
    ax.set_ylabel("m/s", color="white", fontsize=8)
    _legend(ax)

    ax = axes[1]
    ax.set_title("Gap vs Desired Gap", color="white", fontsize=9)
    _plot(ax, t, data["gap01"], "gap01", PAIR_COLORS["01"])
    _plot(ax, t, data["des01"], "des01", PAIR_COLORS["01"], "--", 1.0)
    _plot(ax, t, data["gap12"], "gap12", PAIR_COLORS["12"])
    _plot(ax, t, data["des12"], "des12", PAIR_COLORS["12"], "--", 1.0)
    ax.axhline(D_MIN, color="#e57373", linewidth=0.9, linestyle=":", label="d_min")
    ax.set_ylabel("m", color="white", fontsize=8)
    _legend(ax)

    ax = axes[2]
    ax.set_title("Spacing Error", color="white", fontsize=9)
    _plot(ax, t, data["es01"], "e_s01", PAIR_COLORS["01"])
    _plot(ax, t, data["es12"], "e_s12", PAIR_COLORS["12"])
    ax.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_ylabel("m", color="white", fontsize=8)
    _legend(ax)

    ax = axes[3]
    ax.set_title("Relative Speed Error", color="white", fontsize=9)
    _plot(ax, t, data["ev01"], "e_v01", PAIR_COLORS["01"])
    _plot(ax, t, data["ev12"], "e_v12", PAIR_COLORS["12"])
    ax.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_ylabel("m/s", color="white", fontsize=8)
    _legend(ax)

    ax = axes[4]
    ax.set_title("Delay and DTH", color="white", fontsize=9)
    _plot(ax, t, data["tau01"], "tau01", PAIR_COLORS["01"])
    _plot(ax, t, data["tau12"], "tau12", PAIR_COLORS["12"])
    _plot(ax, t, data["h1"], "h1", "#81c784", "--", 1.0)
    _plot(ax, t, data["h2"], "h2", "#64b5f6", "--", 1.0)
    ax.set_ylabel("s", color="white", fontsize=8)
    _legend(ax)

    ax = axes[5]
    ax.set_title("Tube Radius and Trigger", color="white", fontsize=9)
    _plot(ax, t, data["omega_s1"], "omega_s1", PAIR_COLORS["01"])
    _plot(ax, t, data["omega_s2"], "omega_s2", PAIR_COLORS["12"])
    _plot(ax, t, data["triggered1"], "trigger1", "#ffee58", "--", 1.0)
    _plot(ax, t, data["triggered2"], "trigger2", "#a5d6a7", "--", 1.0)
    _legend(ax)

    ax = axes[6]
    ax.set_title("Longitudinal Command", color="white", fontsize=9)
    _plot(ax, t, data["a0"], "a0", VEHICLE_COLORS["vehicle_0"])
    _plot(ax, t, data["a1"], "a1", VEHICLE_COLORS["vehicle_1"])
    _plot(ax, t, data["a2"], "a2", VEHICLE_COLORS["vehicle_2"])
    ax.axhline(1.5, color="#e57373", linewidth=0.8, linestyle=":")
    ax.axhline(-2.0, color="#e57373", linewidth=0.8, linestyle=":")
    ax.set_ylabel("m/s^2", color="white", fontsize=8)
    _legend(ax)

    ax = axes[7]
    ax.set_title("Trajectory Map", color="white", fontsize=9)
    ax.set_aspect("equal")
    _plot(ax, ref_x, ref_y, "ref path", "#ff9800", "--", 1.0)
    _plot(ax, data["x0"], data["y0"], "vehicle_0", VEHICLE_COLORS["vehicle_0"])
    _plot(ax, data["x1"], data["y1"], "vehicle_1", VEHICLE_COLORS["vehicle_1"])
    _plot(ax, data["x2"], data["y2"], "vehicle_2", VEHICLE_COLORS["vehicle_2"])
    for vid, x_key, y_key in [
        ("vehicle_0", "x0", "y0"),
        ("vehicle_1", "x1", "y1"),
        ("vehicle_2", "x2", "y2"),
    ]:
        xs = _finite(data[x_key])
        ys = _finite(data[y_key])
        if xs and ys:
            ax.plot(xs[-1], ys[-1], "o", color=VEHICLE_COLORS[vid], markersize=5)
    _legend(ax)

    title = f"Platoon Experiment Dashboard - Route {route}"
    if offline and len(t) > 0:
        title += "\n" + format_summary(summarize(data))
    fig.suptitle(title, color="white", fontsize=10)
    fig.tight_layout(pad=2.0)
    fig.canvas.draw_idle()


def load_csv_data(csv_path):
    data = _empty_plot_data()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            _append_row(data, row)
    return data


def parse_args(argv):
    route = "A"
    csv_path = None
    for arg in argv:
        if arg.startswith("route:="):
            route = arg.split(":=")[1].upper()
        elif arg.startswith("csv:="):
            csv_path = arg.split(":=", 1)[1]
    return route, csv_path


def main(args=None):
    argv = sys.argv[1:] if args is None else args
    route, csv_path = parse_args(argv)

    pkg_share = get_package_share_directory("carla_platoon_control")
    route_csv_path = os.path.join(pkg_share, "config", f"route_{route}.csv")
    ref_x, ref_y = [], []
    with open(route_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref_x.append(float(row["x"]))
            ref_y.append(float(row["y"]))

    if csv_path:
        data = load_csv_data(csv_path)
        summary = summarize(data)
        print(format_summary(summary))
        fig, axes = build_figure()
        draw_dashboard(fig, axes, data, ref_x, ref_y, route, offline=True)
        plt.show()
        return

    rclpy.init(args=args)
    node = Visualizer(ref_x, ref_y, route)

    fig, axes = build_figure()

    def update(_):
        rclpy.spin_once(node, timeout_sec=0)
        if len(node.data["t"]) == 0:
            return
        draw_dashboard(fig, axes, node.data, ref_x, ref_y, route, offline=False)

    ani = animation.FuncAnimation(fig, update, interval=100, cache_frame_data=False)
    node.ani = ani
    plt.show()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
