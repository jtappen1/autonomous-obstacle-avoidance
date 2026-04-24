#pragma once

#include <mutex>
#include <optional>

#include <Eigen/Dense>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/mat.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

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
    void depthCb(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
    void infoCb(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg);
    void detectionCb(const vision_msgs::msg::Detection2DArray::ConstSharedPtr& msg);

    void publishMarkers(const std::vector<Obstacle>& obstacles,
                        const std::string& frame_id);
    void publishDeleteMarkers(const std::vector<int>& dead_ids,
                              const std::string& frame_id);

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
    rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr det_sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

    std::mutex depth_mutex_;
    cv::Mat depth_image_;
    std::optional<Intrinsics> intrinsics_;
    std::optional<rclcpp::Time> last_update_stamp_;

    TrackManager2D tracker_;
    tf2_ros::Buffer tf_buffer_{get_clock()};
    tf2_ros::TransformListener tf_listener_{tf_buffer_};
};

}  // namespace final_project::tracker