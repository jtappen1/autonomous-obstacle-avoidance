# Multi-Sensor Autonomous Driving for Obstacle Avoidance and Overtaking under Limited Computation

## 3D Multi-Object Tracker

A ROS 2 package for real-time 3D tracking of people, balls, and cars using a depth camera and YOLO-based detection.

### Overview

Detects people in an RGB stream with a YOLOv8 model, lifts the 2D bounding boxes to 3D using aligned depth, and tracks them over time with per-object Kalman filters. Trajectories are published as `MarkerArray` messages for visualization in RViz/Foxglove.

### Architecture

```
/camera/...  (RGB + Depth)
       │
       ▼
 detections.py          — YOLOv26n inference → Detection2DArray
       │
       ▼
 tracker_node.cpp       — ROS2 node; lifts detections to 3D via depth, runs the multi-tracker
       │
       ▼
 multi_tracker.cpp      — predict / match / update / prune loop
       │
       ▼
 kalman_filter_3d.cpp   — constant-velocity EKF per track
       │
       ▼
/tracker/markers        — MarkerArray (line strips + spheres)
```

### Components

##### `detections.py`
Subscribes to the color image, runs a quantized fp16 YOLOv26n, and publishes `Detection2DArray` on `/detector/detections`.

#### `tracker_node.cpp`
Receives detections and the aligned depth image. For each detection it samples a 5×5 patch of depth, takes the median, and back-projects to camera-frame 3D coordinates `(z, x, -y)`. Calls `MultiTracker3D::step()` each frame and publishes RViz markers.

#### `multi_tracker.cpp`
Greedy nearest-neighbor data association using Mahalanobis distance with a configurable gating threshold. Spawns new tracks for unmatched measurements, prunes tracks that have been missed for too many consecutive frames.

#### `kalman_filter_3d.cpp`
Constant-velocity Kalman filter. State: `[x, y, z, vx, vy, vz]`. Measurement: `[x, y, z]`. Uses Joseph-form covariance update for numerical stability.

### Topics

| Topic | Type | Direction |
|---|---|---|
| `/camera/camera/color/image_raw` | `sensor_msgs/Image` | in |
| `/camera/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | in |
| `/camera/camera/aligned_depth_to_color/camera_info` | `sensor_msgs/CameraInfo` | in |
| `/detector/detections` | `vision_msgs/Detection2DArray` | in/out |
| `/tracker/markers` | `visualization_msgs/MarkerArray` | out |

### Dependencies

- ROS 2 (tested on Humble)
- `sensor_msgs`, `vision_msgs`, `visualization_msgs`, `geometry_msgs`
- Eigen3
- OpenCV + `cv_bridge`
- Python: `ultralytics`, `rclpy`, `cv_bridge`, `numpy`

### Running
```bash
./run.sh
```

Visualize tracks in RViz by adding a `MarkerArray` display on `/tracker/markers`.