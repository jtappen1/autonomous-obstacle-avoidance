ros2 launch realsense2_camera rs_launch.py \
  depth_module.depth_profile:=640x480x30 \
  rgb_camera.color_profile:=640x480x30 \
  pointcloud.enable:=true \
  align_depth.enable:=true