#pragma once
#include "final_project/tracker/multi_tracker.hpp"

#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <optional>
#include <mutex>
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
        const std::string& frame_id = "camera_link");
    
    void publishDeleteMarkers(const std::vector<int>& dead_ids,
                        const std::string& frame_id = "camera_link");

    // ── Subscriptions & publisher ──────────────────────────────────────────
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr       depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr  info_sub_;
    rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr det_sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

    // ── State ──────────────────────────────────────────────────────────────
    std::mutex              depth_mutex_;
    cv::Mat                 depth_image_;          ///< Latest depth frame
    std::optional<Intrinsics> intrinsics_;

    MultiTracker3D tracker_;

};