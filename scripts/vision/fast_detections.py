#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from ultralytics import YOLO
import threading
import cv2
import queue
import time

class DetectionsNode(Node):
    def __init__(self):
        super().__init__('detections_node')
        
        self.pub_detections = self.create_publisher(
            Detection2DArray, '/detector/detections', 10)
        
        # 1. Load TensorRT Engine (Ensure .engine is used for Jetson)
        self.model = YOLO(
            '/home/nvidia/ros2_ws/src/final_project/models/f1tenth_fp16.engine',
            task='detect'
        )

        # 2. GStreamer Pipeline: Much faster than simple cv2.VideoCapture on Jetson
        # This pipeline pulls raw frames and converts them to BGR in hardware/low-level
        gst_pipeline = (
            "v4l2src device=/dev/video4 ! "
            "video/x-raw, width=640, height=480, framerate=60/1 ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! appsink drop=1"
        )
        
        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            self.get_logger().error("GStreamer pipeline failed. Check device path or drivers.")
            return

        # 3. Threading setup
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = True
        
        # Start the producer thread (Camera Reader)
        self.reader_thread = threading.Thread(target=self._camera_reader, daemon=True)
        self.reader_thread.start()

        self.get_logger().info("Detections Node Started with GStreamer and Threading")

    def _camera_reader(self):
        """Dedicated thread to keep the camera buffer empty and provide latest frames."""
        while self.running and rclpy.ok():
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Keep only the freshest frame in the queue
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

    def run_inference(self):
        """Main loop that pulls from the queue and runs YOLO."""
        while rclpy.ok():
            try:
                # Wait for a frame with a timeout so we can check rclpy.ok()
                frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # 4. Optimized Inference: 
            # imgsz=640 (int) is faster; stream=True uses a generator for better memory management
            results = self.model.predict(
                frame, 
                verbose=False, 
                classes=[0], 
                imgsz=640, 
                stream=True,
                device=0
            )

            array_msg = Detection2DArray()
            # Synchronize timestamp with the exact moment of inference
            array_msg.header.stamp = self.get_clock().now().to_msg()
            array_msg.header.frame_id = "camera"

            for result in results:
                for box in result.boxes:
                    det = Detection2D()
                    det.header = array_msg.header

                    # Convert TensorRT output to list once
                    xywh = box.xywh[0].tolist()
                    
                    bbox = BoundingBox2D()
                    bbox.center.position.x = xywh[0]
                    bbox.center.position.y = xywh[1]
                    bbox.size_x = xywh[2]
                    bbox.size_y = xywh[3]
                    det.bbox = bbox

                    hyp = ObjectHypothesisWithPose()
                    hyp.hypothesis.score = float(box.conf[0])
                    det.results.append(hyp)
                    array_msg.detections.append(det)

            self.pub_detections.publish(array_msg)

    def stop(self):
        self.running = False
        if self.reader_thread.is_alive():
            self.reader_thread.join()
        self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    node = DetectionsNode()
    
    try:
        # We don't use rclpy.spin(node) here because the loop is blocking.
        # Instead, we run the custom loop. 
        node.run_inference()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()