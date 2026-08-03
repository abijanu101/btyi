import numpy as np
from typing import Tuple 

class NSAKalman:
    """
        Attributes:
            x(np.array): State Column-Vector
            P(np.array): Covariance Matrix
            H(np.array): Observation Matrix
            F(np.array): State Transition Matrix
            Q(np.array): Process Noise
            R(np.array): Measurement Noise

        No control vectors required for MOT, so no B matrix either
    """

    def __init__(self, x:int, y:int, w:int, h:int):
        self.x = np.array([x, y, w, h, *[0]*4], dtype=float)

        self.H = np.zeros((4, 4*2))
        self.H[:4, :4] = np.eye(4)      # velocities hidden

        self.F = np.eye(8)
        for i in range(4):              # velocity additions in idealistic model
            self.F[i, 4+i] = 1

        self.R = np.eye(4, dtype=float) * 10     # measurement noise
        self.Q = np.eye(8, dtype=float) * 10     # process noise
        self.Q[4:, 4:] *= 10

        self.P = np.eye(8, dtype=float) * 10     # state covariance
        self.P[4:, 4:] *= 100

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x, self.P

    def fuse(self, x:int, y:int, w:int, h:int, conf:float) -> Tuple[np.ndarray, np.ndarray]:
        z = np.array([x, y, w, h], dtype=float)
        res = z - self.H @ self.x                   # pre-fit residual

        R_NSA = (1 - conf + 1e-6) * self.R          # the NSA part with some numerical stability
        S = self.H @ self.P @ self.H.T + R_NSA      # residual cov
        K = self.P @ self.H.T @ np.linalg.inv(S)    # optimal Kalman Gain

        self.x = self.x + K @ res
        self.P = (np.eye(8) - K @ self.H) @ self.P

        return self.x, self.P
