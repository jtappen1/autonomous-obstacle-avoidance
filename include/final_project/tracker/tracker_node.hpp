#pragma once
#include "final_project/tracker/multi_tracker.hpp"

#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <overtake_msgs/msg/tracked_obstacle.hpp>
#include <overtake_msgs/msg/tracked_obstacle_array.hpp>

#include <optional>
#include <memory>
#include <mutex>
#include <string>
#include <opencv2/core/mat.hpp>
#include <Eigen/Dense>

struct Intrinsics {
    double fx, fy, cx, cy;
};

class TrackerNode : public rclcpp::Node {
public:
    TrackerNode();

private:
    // ── Callbacks ──────────────────────────────────────────────────────────
    void depthCb(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
    void infoCb(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg);
    void detectionCb(
        const vision_msgs::msg::Detection2DArray::ConstSharedPtr& msg);

    // ── Helpers ────────────────────────────────────────────────────────────
    void publishMarkers(
        const std::vector<Obstacle>& obstacles,
        const std::string& frame_id);

    void publishDeleteMarkers(const std::vector<int>& dead_ids,
                        const std::string& frame_id);

    // Publish the globally tracked obstacle state for the planner.
    void publishObstacles(
        const std::vector<Obstacle>& obstacles,
        const std::vector<int>& dead_ids,
        const builtin_interfaces::msg::Time& stamp);

    // ── Subscriptions & publisher ──────────────────────────────────────────
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr       depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr  info_sub_;
    rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr det_sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<overtake_msgs::msg::TrackedObstacleArray>::SharedPtr obstacle_pub_;

    // ── TF ─────────────────────────────────────────────────────────────────
    std::unique_ptr<tf2_ros::Buffer>            tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

    // ── Params ─────────────────────────────────────────────────────────────
    std::string global_frame_;    // e.g. "map"
    std::string sensor_frame_;    // fallback source frame when message headers are empty
    std::string obstacle_topic_;  // e.g. "/overtake/obstacles"

    // ── State ──────────────────────────────────────────────────────────────
    std::mutex              depth_mutex_;
    cv::Mat                 depth_image_;          ///< Latest depth frame
    std::string             depth_frame_id_;
    std::string             camera_info_frame_id_;
    std::optional<builtin_interfaces::msg::Time> depth_stamp_;
    std::optional<Intrinsics> intrinsics_;

    MultiTracker3D tracker_;

};
