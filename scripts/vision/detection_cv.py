#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from ultralytics import YOLO
import threading
import cv2


class Detections(Node):

    def __init__(self):
        super().__init__('detections')
        
        self.pub_detections = self.create_publisher(
            Detection2DArray, '/detector/detections', 10)
        
        self.model = YOLO(
            '/home/nvidia/ros2_ws/src/final_project/models/f1tenth_fp16.engine',
            task='detect')

        self.cap = cv2.VideoCapture('/dev/video4', cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera")
            return

    def infer_loop(self):
        while rclpy.ok():
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn("Failed to read frame")
                continue

            results = self.model(frame, verbose=False, classes=[0], imgsz=(640, 480), device=0)

            array_msg = Detection2DArray()
            array_msg.header.stamp = self.get_clock().now().to_msg()
            array_msg.header.frame_id = "camera"

            for box in results[0].boxes:
                det = Detection2D()
                det.header = array_msg.header

                bbox = BoundingBox2D()
                bbox.center.position.x, bbox.center.position.y = box.xywh[0][:2].tolist()
                bbox.size_x, bbox.size_y = box.xywh[0][2:].tolist()
                det.bbox = bbox

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.score = float(box.conf[0])
                det.results.append(hyp)
                array_msg.detections.append(det)

            self.pub_detections.publish(array_msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Detections()
    node.infer_loop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()