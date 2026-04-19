#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import cv2
from ultralytics import YOLO
import time
import torch

class Detect3D(Node):
    def __init__(self):
        super().__init__('detect_3d')
        self.bridge = CvBridge()
        self.depth_image = None
        self.intrinsics = None  # (fx, fy, cx, cy)
        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_cb, 10)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.info_cb, 10)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.rgb_cb, 10)

        self.image_pub = self.create_publisher(Image, '/final_project/image', 10)
        self.model = YOLO('/home/nvidia/ros2_ws/src/final_project/models/yolo26n_fast_fp16.engine')
        print(f"Loaded model")
        self.prev = time.time()

    def info_cb(self, msg):
        # K matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self.intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])  # fx, fy, cx, cy

    def depth_cb(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def rgb_cb(self, msg):
        if self.depth_image is None or self.intrinsics is None:
            return
        rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        # --- Run your YOLO model here ---
        t0 = time.time()
        #results = self.model.track(rgb, verbose=False, classes=[0], imgsz=(640, 480))  # returns list of (u, v, w, h)
        results = self.model.track(rgb, persist=True, tracker="botsort.yaml", verbose=False, classes=[0], imgsz=(640, 480))
        latency_ms = (time.time() - t0) * 1000


        annotated = results[0].plot()

        now = time.time()
        fps = 1.0 / (now - self.prev)
        self.prev = now

        h, w = annotated.shape[:2]
        cv2.putText(annotated, f'Size: {w}x{h}', (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(annotated, f"Latency: {latency_ms:.1f}ms", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        
        out_msg = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
        out_msg.header = msg.header
        self.image_pub.publish(out_msg)

        for result in results:
            self.boxes_to_3d(result.boxes, self.depth_image)

    def boxes_to_3d(self, boxes, depth_image):
        """
        boxes: Ultralytics Boxes object
        depth_image: (H, W) numpy array
        """

        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 3))

        # extract xyxy
        xyxy = boxes.xyxy
        if boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = np.full(len(boxes), -1)
        depth_image = depth_image.astype(np.float32) * 0.001

        # ensure numpy
        if hasattr(xyxy, "cpu"):
            xyxy = xyxy.cpu().numpy()

        x1 = xyxy[:, 0]
        y1 = xyxy[:, 1]
        x2 = xyxy[:, 2]
        y2 = xyxy[:, 3]

        cx = ((x1 + x2) * 0.5).astype(np.int32)
        cy = ((y1 + y2) * 0.5).astype(np.int32)

        H, W = depth_image.shape

        cx = np.clip(cx, 0, W - 1)
        cy = np.clip(cy, 0, H - 1)

        Z = depth_image[cy, cx].astype(np.float32)

        valid = Z > 0
        cx = cx[valid]
        cy = cy[valid]
        Z = Z[valid]
        valid_ids = track_ids[valid]
        if len(Z) == 0:
            return np.zeros((0, 3))

        fx, fy, cx0, cy0 = self.intrinsics

        X = (cx - cx0) * Z / fx
        Y = (cy - cy0) * Z / fy
        #print(f"X: {X}, Y: {Y}, Z: {Z}")
        #print(depth_image.dtype, depth_image.min(), depth_image.max())
        for tid, x_val, y_val, z_val in zip(valid_ids, X, Y, Z):
            print(f"ID: {tid} | X: {x_val:.3f}, Y: {y_val:.3f}, Z: {z_val:.3f}")
        return np.stack([X, Y, Z], axis=1), valid_ids

def main(args=None):
    rclpy.init(args=args)
    node = Detect3D()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
