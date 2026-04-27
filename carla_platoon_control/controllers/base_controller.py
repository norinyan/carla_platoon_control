from abc import ABC, abstractmethod


class LateralController(ABC):
    @abstractmethod
    def compute(self, vehicle_state, ref_points, dt):
        """
        横向控制器接口

        vehicle_state: 当前车辆状态，包含 x, y, yaw, speed
        ref_points: 参考轨迹点列表，每个点包含 x, y, yaw, kappa, speed, s
        dt: 控制周期，单位秒

        return: steer，范围 [-1.0, 1.0]
        """
        raise NotImplementedError


class LeaderLonController(ABC):
    @abstractmethod
    def compute(self, vehicle_state, ref_points, dt):
        """
        领航车纵向控制器接口

        vehicle_state: 当前车辆状态，包含 x, y, yaw, speed, s
        ref_points: 参考轨迹点列表，每个点包含 x, y, yaw, kappa, speed, s
        dt: 控制周期，单位秒

        return: a_cmd，单位 m/s^2
        """
        raise NotImplementedError


class FollowerLonController(ABC):
    @abstractmethod
    def compute(self, vehicle_state, ref_points, front_state, desired_gap, dt):
        """
        跟随车纵向控制器接口

        vehicle_state: 当前车辆状态，包含 x, y, yaw, speed, s
        ref_points: 参考轨迹点列表，每个点包含 x, y, yaw, kappa, speed, s
        front_state: 前车状态，包含 x, y, yaw, speed, s，可以额外包含 accel
        desired_gap: 期望跟车净间距，单位米
        dt: 控制周期，单位秒

        return: a_cmd，单位 m/s^2
        """
        raise NotImplementedError
