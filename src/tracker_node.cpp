#include "final_project/tracker/tracker_node.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace final_project::tracker {

using std::placeholders::_1;

namespace {

double medianInPlace(std::vector<float>& values) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
    return static_cast<double>(values[mid]);
}

std::string resolveFrameId(const std::string& header_frame_id) {
    return header_frame_id.empty() ? std::string("camera_link") : header_frame_id;
}

}  // namespace

TrackerNode::TrackerNode()
    : Node("tracker_node"),
      tracker_(1.0 / 30.0) {
    depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/camera/camera/aligned_depth_to_color/image_raw", 10,
        std::bind(&TrackerNode::depthCb, this, _1));

    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        "/camera/camera/aligned_depth_to_color/camera_info", 10,
        std::bind(&TrackerNode::infoCb, this, _1));

    det_sub_ = create_subscription<vision_msgs::msg::Detection2DArray>(
        "/detector/detections", 10,
        std::bind(&TrackerNode::detectionCb, this, _1));

    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        "/tracker/markers", 10);

    RCLCPP_INFO(get_logger(), "TrackerNode initialized");
}

void TrackerNode::infoCb(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg) {
    intrinsics_ = Intrinsics{msg->k[0], msg->k[4], msg->k[2], msg->k[5]};
}

void TrackerNode::depthCb(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    cv::Mat depth;
    try {
        depth = cv_bridge::toCvCopy(msg, msg->encoding)->image;
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_WARN(get_logger(), "cv_bridge: %s", e.what());
        return;
    }

    if (depth.type() == CV_32FC1) {
        depth.convertTo(depth, CV_16UC1, 1000.0);
    }

    std::lock_guard<std::mutex> lock(depth_mutex_);
    depth_image_ = std::move(depth);
}

void TrackerNode::detectionCb(const vision_msgs::msg::Detection2DArray::ConstSharedPtr& msg) {
    if (!intrinsics_) {
        RCLCPP_WARN(get_logger(), "detectionCb: no intrinsics yet, dropping frame");
        return;
    }

    cv::Mat depth;
    {
        std::lock_guard<std::mutex> lock(depth_mutex_);
        if (depth_image_.empty()) {
            RCLCPP_WARN(get_logger(), "detectionCb: depth image empty, dropping frame");
            return;
        }
        depth = depth_image_;
    }

    const auto [fx, fy, cx0, cy0] = *intrinsics_;
    const int img_rows = depth.rows;
    const int img_cols = depth.cols;

    std::vector<DetectionMeasurement> measurements;
    measurements.reserve(msg->detections.size());

    for (const auto& det : msg->detections) {
        const int u = static_cast<int>(det.bbox.center.position.x);
        const int v = static_cast<int>(det.bbox.center.position.y);

        const int x1 = std::max(u - 2, 0);
        const int x2 = std::min(u + 3, img_cols);
        const int y1 = std::max(v - 2, 0);
        const int y2 = std::min(v + 3, img_rows);
        if (x1 >= x2 || y1 >= y2) continue;

        const cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
        cv::Mat patch;
        depth(roi).convertTo(patch, CV_32F);

        std::vector<float> valid_depths;
        valid_depths.reserve(patch.total());
        for (int r = 0; r < patch.rows; ++r) {
            const float* row = patch.ptr<float>(r);
            for (int c = 0; c < patch.cols; ++c) {
                if (row[c] > 0.f) valid_depths.push_back(row[c]);
            }
        }
        if (valid_depths.empty()) continue;

        std::vector<float> depths_for_median = valid_depths;
        const double z = medianInPlace(depths_for_median) * 0.001;

        std::vector<float> abs_dev;
        abs_dev.reserve(valid_depths.size());
        for (float d_mm : valid_depths) {
            abs_dev.push_back(std::fabs(d_mm * 0.001f - static_cast<float>(z)));
        }
        const double mad = medianInPlace(abs_dev);

        const double x = (u - cx0) * z / fx;
        const double y = (v - cy0) * z / fy;

        DetectionMeasurement measurement;
        measurement.pos = Eigen::Vector2d(z, x);   // [forward, lateral]
        measurement.height = -y;

        const double sigma_forward = std::max(0.08, 1.4826 * mad + 0.01 * z);
        const double sigma_lateral = std::max(0.05, (z / std::max(fx, 1.0)) * 3.0 + 0.25 * sigma_forward);
        measurement.R << sigma_forward * sigma_forward, 0.0,
                         0.0, sigma_lateral * sigma_lateral;
        measurements.push_back(measurement);
    }

    rclcpp::Time stamp = msg->header.stamp.sec == 0 && msg->header.stamp.nanosec == 0
                             ? now()
                             : rclcpp::Time(msg->header.stamp);
    double dt = 1.0 / 30.0;
    if (last_update_stamp_) {
        dt = (stamp - *last_update_stamp_).seconds();
        if (!std::isfinite(dt) || dt <= 0.0 || dt > 0.25) {
            dt = 1.0 / 30.0;
        }
    }
    last_update_stamp_ = stamp;

    const auto result = tracker_.step(measurements, dt);
    const std::string frame_id = resolveFrameId(msg->header.frame_id);
    publishMarkers(result.obstacles, frame_id);
    publishDeleteMarkers(result.dead_ids, frame_id);
}

void TrackerNode::publishMarkers(const std::vector<Obstacle>& obstacles,
                                 const std::string& frame_id) {
    visualization_msgs::msg::MarkerArray array;
    auto now_time = get_clock()->now();

    for (const auto& ob : obstacles) {
        visualization_msgs::msg::Marker history;
        history.header.frame_id = frame_id;
        history.header.stamp = now_time;
        history.ns = "tracks";
        history.id = ob.id;
        history.type = visualization_msgs::msg::Marker::LINE_STRIP;
        history.action = visualization_msgs::msg::Marker::ADD;
        history.scale.x = 0.05;

        std::mt19937 gen(ob.id);
        std::uniform_real_distribution<float> dist(0.2f, 1.0f);
        history.color.r = dist(gen);
        history.color.g = dist(gen);
        history.color.b = dist(gen);
        history.color.a = 1.0f;

        for (const auto& p : ob.history) {
            geometry_msgs::msg::Point pt;
            pt.x = p.x();
            pt.y = p.y();
            pt.z = p.z();
            history.points.push_back(pt);
        }
        array.markers.push_back(history);

        visualization_msgs::msg::Marker current;
        current.header.frame_id = frame_id;
        current.header.stamp = now_time;
        current.ns = "tracks_current";
        current.id = ob.id;
        current.type = visualization_msgs::msg::Marker::SPHERE;
        current.action = visualization_msgs::msg::Marker::ADD;
        current.pose.position.x = ob.position.x();
        current.pose.position.y = ob.position.y();
        current.pose.position.z = ob.position.z();
        current.pose.orientation.w = 1.0;
        current.scale.x = 0.12;
        current.scale.y = 0.12;
        current.scale.z = 0.12;
        current.color.r = 1.0f;
        current.color.g = 0.2f;
        current.color.b = 0.2f;
        current.color.a = 1.0f;
        array.markers.push_back(current);

        visualization_msgs::msg::Marker pred;
        pred.header.frame_id = frame_id;
        pred.header.stamp = now_time;
        pred.ns = "tracks_predicted";
        pred.id = ob.id;
        pred.type = visualization_msgs::msg::Marker::LINE_STRIP;
        pred.action = visualization_msgs::msg::Marker::ADD;
        pred.scale.x = 0.03;
        pred.color.r = history.color.r;
        pred.color.g = history.color.g;
        pred.color.b = history.color.b;
        pred.color.a = 0.4f;
        for (const auto& p : ob.predicted_trajectory) {
            geometry_msgs::msg::Point pt;
            pt.x = p.position.x();
            pt.y = p.position.y();
            pt.z = p.position.z();
            pred.points.push_back(pt);
        }
        array.markers.push_back(pred);
    }

    marker_pub_->publish(array);
}

void TrackerNode::publishDeleteMarkers(const std::vector<int>& dead_ids,
                                       const std::string& frame_id) {
    if (dead_ids.empty()) return;

    visualization_msgs::msg::MarkerArray array;
    auto now_time = get_clock()->now();

    for (int id : dead_ids) {
        visualization_msgs::msg::Marker m;
        m.header.stamp = now_time;
        m.header.frame_id = frame_id;
        m.ns = "tracks";
        m.id = id;
        m.action = visualization_msgs::msg::Marker::DELETE;
        array.markers.push_back(m);

        visualization_msgs::msg::Marker curr;
        curr.header.stamp = now_time;
        curr.header.frame_id = frame_id;
        curr.ns = "tracks_current";
        curr.id = id;
        curr.action = visualization_msgs::msg::Marker::DELETE;
        array.markers.push_back(curr);

        visualization_msgs::msg::Marker pred;
        pred.header.stamp = now_time;
        pred.header.frame_id = frame_id;
        pred.ns = "tracks_predicted";
        pred.id = id;
        pred.action = visualization_msgs::msg::Marker::DELETE;
        array.markers.push_back(pred);
    }

    marker_pub_->publish(array);
}

}  // namespace final_project::tracker