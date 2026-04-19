#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from vision_msgs.msg import Detection2DArray, Detection2D
from std_msgs.msg import Header
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

        # ensure numpy
        if measurement is not None:
            measurement = np.asarray(measurement).reshape(3)

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

class Tracker(Node):

    def __init__(self):
        super().__init__('tracker')

        self.bridge = CvBridge()

        self.create_subscription(
            Detection2DArray,
            '/detector/detections',
            self.detection_cb,
            10
        )

        self.create_subscription(
            CameraInfo,
            '/camera/camera/aligned_depth_to_color/camera_info',
            self.info_cb,
            10
        )

        self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_cb,
            10
        )

        self.pub_cloud = self.create_publisher(PointCloud2, '/tracker/points', 10)

        self.intrinsics = None
        self.depth_image = None

        self.tracker = Track3D(dt= 1.0 / 30.0)

        print("Init Tracker")

    def publish_pointcloud(self, points, frame_id="camera_link"):

        if points is None or len(points) == 0:
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id

        cloud_msg = pc2.create_cloud_xyz32(header, points.tolist())
        self.pub_cloud.publish(cloud_msg)

    def info_cb(self, msg):
        fx = msg.k[0]
        fy = msg.k[4]
        cx = msg.k[2]
        cy = msg.k[5]
        self.intrinsics = (fx, fy, cx, cy)

    # -------------------------
    # Depth image
    # -------------------------
    def depth_cb(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='passthrough'
        )

    # -------------------------
    # Detection callback
    # -------------------------
    def detection_cb(self, msg):
        if self.depth_image is None or self.intrinsics is None:
            return

        fx, fy, cx0, cy0 = self.intrinsics
        points_3d = []

        for detection in msg.detections:

            bbox = detection.bbox
            center = bbox.center

            u = int(center.position.x)
            v = int(center.position.y)

            x1 = max(u - 2, 0)
            x2 = min(u + 3, self.depth_image.shape[1])
            y1 = max(v - 2, 0)
            y2 = min(v + 3, self.depth_image.shape[0])

            depth_patch = self.depth_image[y1:y2, x1:x2].astype(np.float32)

            valid = depth_patch > 0
            if np.count_nonzero(valid) == 0:
                continue

            z = np.median(depth_patch[valid]) * 0.001

            x = (u - cx0) * z / fx
            y = (v - cy0) * z / fy

            points_3d.append([z, x, -y])

        if len(points_3d) == 0:
            # no detection → just predict
            pos, pred, hist = self.tracker.step(None)
            return
        else:
            # assume single person for now
            measurement = points_3d[0]
            pos, pred, hist = self.tracker.step(measurement)

        points_3d = np.array(points_3d)

        # print("3D detections:\n", points_3d)

        viz_points = []

        # history points
        for p in hist:
            viz_points.append(p)

        # predicted point
        viz_points.append(pred)

        viz_points = np.array(viz_points)

        self.publish_pointcloud(viz_points)



def main(args=None):
    rclpy.init(args=args)
    node = Tracker()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()