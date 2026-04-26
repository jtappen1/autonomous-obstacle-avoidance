"""Launch the YOLO detector and the LiDAR/camera fusion tracker.

The tracker publishes overtake_msgs/TrackedObstacleArray on
/overtake/tracked_obstacles, which is the default obstacle_topic that the
MPC overtake planner (lab-8-model-predictive-control-team4/mpc) subscribes
to. Bring this launch up alongside the MPC node and (in sim) make sure the
slow_dynamic_obstacle_node is NOT also publishing to the same topic.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_fast_detector = DeclareLaunchArgument(
        'use_fast_detector',
        default_value='true',
        description='true: GStreamer/V4L2 direct-capture detector (real car); '
                    'false: subscribe-to-/camera/* detector (sim or RealSense node).',
    )

    fast_detector = Node(
        package='final_project',
        executable='fast_detections.py',
        name='detections',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_fast_detector')),
    )

    ros_detector = Node(
        package='final_project',
        executable='detections.py',
        name='detections',
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('use_fast_detector')),
    )

    tracker = Node(
        package='final_project',
        executable='tracker_node',
        name='tracker_node',
        output='screen',
    )

    return LaunchDescription([
        use_fast_detector,
        fast_detector,
        ros_detector,
        tracker,
    ])
