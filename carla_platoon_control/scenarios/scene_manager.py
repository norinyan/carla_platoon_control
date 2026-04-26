import sys
import os
import carla
import yaml
import csv
import math

sys.path.append(os.path.expanduser('~/carla/PythonAPI/carla'))

_SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
_CONFIG_DIR     = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', 'config'))
_DEFAULT_PARAMS = os.path.join(_CONFIG_DIR, 'vehicle_params.yaml')
_DEFAULT_CSV_DIR = _CONFIG_DIR

VEHICLE_IDS = ['vehicle_0', 'vehicle_1', 'vehicle_2']

class SceneManager:
    def __init__(self, params_path=_DEFAULT_PARAMS, csv_dir=_DEFAULT_CSV_DIR):
        params_path = os.path.abspath(params_path)
        csv_dir     = os.path.abspath(csv_dir)

        with open(params_path, 'r') as f:
            self.params = yaml.safe_load(f)

        self.csv_dir  = csv_dir
        self.vehicles = {}   # {vehicle_id: actor}
        self.spectator = None

        # 连接CARLA
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)

        scene_cfg   = self.params['scene']
        target_map  = scene_cfg['map']
        current_map = self.client.get_world().get_map().name
        if target_map not in current_map:
            print(f"正在加载地图: {target_map}")
            self.client.load_world(target_map)
        else:
            print(f"地图已是 {target_map}，跳过重新加载")

        self.world = self.client.get_world()

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = scene_cfg['fixed_delta_seconds']
        self.world.apply_settings(settings)

        weather = getattr(carla.WeatherParameters, scene_cfg['weather'])
        self.world.set_weather(weather)

        self.spectator = self.world.get_spectator()

    def load_trajectory(self, route_id='A'):
        csv_path   = os.path.join(self.csv_dir, f'route_{route_id}.csv')
        trajectory = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                trajectory.append({
                    's':     float(row['s']),
                    'x':     float(row['x']),
                    'y':     float(row['y']),
                    'yaw':   float(row['yaw']),
                    'kappa': float(row['kappa']),
                    'speed': float(row['speed']),
                })
        return trajectory

    def spawn_vehicles(self, route_id='A'):
        """在路线起点生成3辆车，纵向排列间距8m（中心距13m）"""
        trajectory = self.load_trajectory(route_id)
        start_pt   = trajectory[0]

        blueprint_library = self.world.get_blueprint_library()
        vehicles_cfg = self.params['vehicles']

        # 3辆车起点：vehicle_0最前，vehicle_2最后
        # CSV是ROS坐标系，y需要取反传给CARLA
        offsets = {
            'vehicle_0': 26.0,   # 最前
            'vehicle_1': 13.0,   # 中间
            'vehicle_2': 0.0,    # 最后
        }

        for vid in VEHICLE_IDS:
            cfg = vehicles_cfg[vid]
            bp  = blueprint_library.find(cfg['blueprint'])
            bp.set_attribute('color', cfg['color'])
            if bp.has_attribute('role_name'):
                bp.set_attribute('role_name', vid)

            offset_x = offsets[vid]
            spawn_transform = carla.Transform(
                carla.Location(
                    x=start_pt['x'] + offset_x,
                    y=start_pt['y'],    # ROS->CARLA y取反
                    z=0.6
                ),
                carla.Rotation(yaw=math.degrees(-start_pt['yaw']))  # yaw取反
            )

            actor = self.world.spawn_actor(bp, spawn_transform)
            self.vehicles[vid] = actor
            print(f"{vid} 已生成于 x={start_pt['x']+offset_x:.2f}")

        return self.vehicles

    def visualize_trajectory(self, route_id='A'):
        trajectory = self.load_trajectory(route_id)
        debug  = self.world.debug
        color  = carla.Color(0, 255, 0)

        for i, pt in enumerate(trajectory):
            loc = carla.Location(x=pt['x'], y=pt['y'], z=0.3)
            if i == 0:
                debug.draw_string(loc, '[A] START', life_time=0, color=color)
                debug.draw_point(loc, size=0.3, life_time=0, color=color)
            elif i == len(trajectory) - 1:
                debug.draw_string(loc, '[A] END', life_time=0, color=color)
                debug.draw_point(loc, size=0.3, life_time=0, color=color)
            else:
                debug.draw_point(loc, size=0.08, life_time=0, color=color)

        print(f"路线A 轨迹已可视化，共{len(trajectory)}个点")

    def set_spectator(self):
        """跟随领航车vehicle_0"""
        if 'vehicle_0' not in self.vehicles:
            return
        transform = self.vehicles['vehicle_0'].get_transform()
        yaw_rad   = math.radians(transform.rotation.yaw)
        offset_x  = -15 * math.cos(yaw_rad)
        offset_y  = -15 * math.sin(yaw_rad)
        self.spectator.set_transform(carla.Transform(
            carla.Location(
                x=transform.location.x + offset_x,
                y=transform.location.y + offset_y,
                z=transform.location.z + 12
            ),
            carla.Rotation(pitch=-35, yaw=transform.rotation.yaw)
        ))

    def tick(self):
        self.world.tick()

    def update_spectator(self):
        self.set_spectator()

    def destroy(self):
        for vid, actor in self.vehicles.items():
            actor.destroy()
            print(f"{vid} 已销毁")
        self.vehicles = {}
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)


if __name__ == '__main__':
    manager = SceneManager()
    manager.spawn_vehicles('A')
    manager.visualize_trajectory('A')
    manager.set_spectator()
    print("场景已就绪，按 Ctrl+C 退出")