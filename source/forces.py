import numpy as np


class Force:
    def __init__(self, body1_index: int, body2_index: int, sensor1_dis: np.ndarray, sensor2_dis: np.ndarray, k: float = 1e6, d: float = 1e4):
        self.body1 = body1_index
        self.body2 = body2_index
        self.sensor1_dis = np.copy(sensor1_dis)
        self.sensor2_dis = np.copy(sensor2_dis)
        self.k = k
        self.d = d

    def get_force(self, pos_sensor1: np.ndarray, pos_sensor2: np.ndarray, vel_sensor1: np.ndarray, vel_sensor2: np.ndarray):
        dpos = pos_sensor1 - pos_sensor2
        dvel = vel_sensor1 - vel_sensor2

        F = -self.k * dpos - self.d * dvel
        return F

    def get_wind_force(self, t):
        return np.array([50.e3 * (t)/50., 0.])

    def get_indices(self):
        return self.body1, self.body2

    def get_dis(self):
        return self.sensor1_dis, self.sensor2_dis
