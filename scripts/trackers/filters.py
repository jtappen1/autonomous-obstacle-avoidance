import numpy as np
from collections import deque


class KalmanFilter3D:

    def __init__(self, dt, sigma_a, sigma_z):
        self.dt = dt

        # State matrix
        self.x = np.zeros((6,1))

        # Covariance Matrix
        self.P = np.eye(6) * 10.0

        # State transition matrix
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Process Noise Covariance Matrix
        self.Q = self._build_process_noise(dt, sigma_a)

        # Measurement Matrix
        self.H = np.zeros((3,6))
        self.H[0,0] = 1
        self.H[1,1] = 1
        self.H[2,2] = 1

        # Measurement Noise matrix
        if np.isscalar(sigma_z):
            self.R = np.eye(3) * sigma_z**2
        else:
            self.R = np.diag(np.array(sigma_z) ** 2)

        # Identity Matrix
        self.I = np.eye(6)
    


    def _build_process_noise(self, dt, sigma_a):
        q = sigma_a**2

        Q_1D = np.array([
            [dt**4 / 4, dt**3 / 2],
            [dt**3 / 2, dt**2]
        ]) * q

        Q = np.zeros((6, 6))

        for i in range(3):
            Q[i, i] = Q_1D[0, 0]
            Q[i, i+3] = Q_1D[0, 1]
            Q[i+3, i] = Q_1D[1, 0]
            Q[i+3, i+3] = Q_1D[1, 1]

        return Q

    def predict(self):
        """
        Predict step, updates the internal state based on state transition matrix and process noise.
        """
        self.x = self.F @ self.x
        self.P = (self.F @ self.P) @ self.F.T + self.Q
        return self.x
    
    def update(self, z):
        """
        z: (3,) or (3,1) measurement [x, y, z]
        """
        z = np.asarray(z).reshape(3, 1)

        # Innovation
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update
        self.P = (self.I - K @ self.H) @ self.P @ (self.I - K @ self.H).T + K @ self.R @ K.T

    def get_position(self):
        return self.x[:3].flatten()

    def get_velocity(self):
        return self.x[3:].flatten()

class Track3D:
    def __init__(self, dt=0.1):
        self.kf = KalmanFilter3D(
            dt=dt,
            sigma_a=1.0,
            sigma_z=[0.1, 0.1, 0.2]
        )

        self.initialized = False

        # store last 10 positions
        self.history = deque(maxlen=10)

    def step(self, measurement):
        """
        measurement: (3,) np array OR None
        """

        # --- predict ---
        self.kf.predict()

        # --- update ---
        if measurement is not None:
            if not self.initialized:
                self.kf.x[:3] = measurement.reshape(3,1)
                self.initialized = True
            else:
                self.kf.update(measurement)

        # --- current state ---
        pos = self.kf.get_position()

        # store history
        self.history.append(pos)

        # --- predict next point (1 step ahead) ---
        next_state = self.kf.F @ self.kf.x
        next_pos = next_state[:3].flatten()

        return pos, next_pos, list(self.history)