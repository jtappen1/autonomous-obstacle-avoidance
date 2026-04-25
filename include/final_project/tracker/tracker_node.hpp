#pragma once

#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>  // CHANGED: LiDAR instead of depth image
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <message_filters/subscriber.h>                       // CHANGED
#include <message_filters/synchronizer.h>                    // CHANGED
#include <message_filters/sync_policies/approximate_time.h>  // CHANGED

#include "final_project/tracker/track_manager.hpp"

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace final_project::tracker {

struct Intrinsics {
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
};

class TrackerNode : public rclcpp::Node {
public:
    TrackerNode();

private:
    // CHANGED: removed depthCb() and the cached depth image members.
    void infoCb(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg);

    // CHANGED: one synchronized callback for detections + LaserScan.
    void fusedCb(const vision_msgs::msg::Detection2DArray::ConstSharedPtr& det_msg,
                 const sensor_msgs::msg::LaserScan::ConstSharedPtr& scan_msg);

    void publishMarkers(const std::vector<Obstacle>& obstacles,
                        const std::string& frame_id);
    void publishDeleteMarkers(const std::vector<int>& dead_ids,
                              const std::string& frame_id);

    // CHANGED: ApproximateTime sync policy for camera detections and LiDAR.
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        vision_msgs::msg::Detection2DArray,
        sensor_msgs::msg::LaserScan>;

    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;

    // CHANGED: message_filters subscribers replace plain rclcpp detection subscription.
    message_filters::Subscriber<vision_msgs::msg::Detection2DArray> det_sub_;
    message_filters::Subscriber<sensor_msgs::msg::LaserScan> scan_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

    std::optional<Intrinsics> intrinsics_;
    std::optional<rclcpp::Time> last_update_stamp_;

    // CHANGED: store the camera optical frame from CameraInfo for projection.
    std::string camera_frame_ = "camera_link";

    // CHANGED: choose your tracking frame here. "odom" is often smoother than "map".
    std::string tracking_frame_ = "map";

    TrackManager2D tracker_;
    tf2_ros::Buffer tf_buffer_{get_clock()};
    tf2_ros::TransformListener tf_listener_{tf_buffer_};

    std::string laser_frame_ = "laser";
};

}  // namespace final_project::tracker