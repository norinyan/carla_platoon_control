from abc import ABC, abstractmethod


class PlatoonController(ABC):
    """
    编队控制器基类，横纵向控制合并在一个接口中。
    """

    @abstractmethod
    def compute(
        self,
        ego_state: dict,
        ref_point: list,
        leader_state: dict,
        desired_gap: float,
        dt: float,
    ) -> tuple:
        """
        ego_state:    本车状态，{x, y, yaw, speed}
        ref_point:    参考轨迹点列表，从当前最近点往前N个点，
                      每个点格式 {x, y, yaw, kappa, speed, s}
        leader_state: 前车状态，{x, y, yaw, speed}
                      领航车(vehicle_0)的 leader_state 传 None
        desired_gap:  期望跟车间距(米)
        dt:           控制周期(秒)

        return: (steer, a_cmd)
            steer:  方向盘控制量，范围 [-1.0, 1.0]
            a_cmd:  纵向加速度指令 (m/s^2)
        """
        pass