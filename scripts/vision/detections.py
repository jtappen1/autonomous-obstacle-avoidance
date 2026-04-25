#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge
import numpy as np
import cv2
from ultralytics import YOLO
import time


class Detections(Node):

    def __init__(self):
        super().__init__('detections')
        self.bridge = CvBridge() 
        self.pub_detections = self.create_publisher(Detection2DArray, '/detector/detections', 10)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.rgb_cb, 10)
        self.model = YOLO('/home/nvidia/ros2_ws/src/final_project/models/f1tenth_fp16.engine')

    def rgb_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        results = results = self.model(frame, verbose=False, classes=[0], imgsz=(640, 480), device=0)
        
        array_msg = Detection2DArray()
        array_msg.header = msg.header  # preserve original timestamp
        
        for box in results[0].boxes:
            det = Detection2D()
            det.header = msg.header
            
            bbox = BoundingBox2D()
            bbox.center.position.x, bbox.center.position.y = box.xywh[0][:2].tolist() 
            bbox.size_x, bbox.size_y = box.xywh[0][2:].tolist()
            det.bbox = bbox

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(int(box.cls[0]))        # class_id as string
            hyp.hypothesis.score = float(box.conf[0])       # confidence
            det.results.append(hyp)

            det.id = str(int(box.id)) if box.id is not None else "-1"
            
            array_msg.detections.append(det)
        
        self.pub_detections.publish(array_msg)

def main(args=None):
    rclpy.init(args=args)
    node = Detections()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ =='__main__':
    main()


