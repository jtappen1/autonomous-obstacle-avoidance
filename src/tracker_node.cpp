#include "final_project/tracker/tracker_node.hpp"

#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/exceptions.h>
#include <opencv2/imgproc.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <cstdint>
#include <limits>
#include <random>


using std::placeholders::_1;

namespace {

bool isOpticalFrame(const std::string& frame_id)
{
    return frame_id.find("optical") != std::string::npos;
}

geometry_msgs::msg::Quaternion yawToQuaternion(double yaw)
{
    geometry_msgs::msg::Quaternion quat;
    const double half = 0.5 * yaw;
    quat.x = 0.0;
    quat.y = 0.0;
    quat.z = std::sin(half);
    quat.w = std::cos(half);
    return quat;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
TrackerNode::TrackerNode()
    : Node("tracker_node"),
      tracker_(1.0 / 30.0)
{
    // ── Params ─────────────────────────────────────────────────────────────
    global_frame_   = this->declare_parameter<std::string>("global_frame",   "map");
    sensor_frame_   = this->declare_parameter<std::string>("sensor_frame",   "camera_link");
    obstacle_topic_ = this->declare_parameter<std::string>("obstacle_topic", "/overtake/tracked_obstacles");

    // ── TF ─────────────────────────────────────────────────────────────────
    tf_buffer_   = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    // ── Subscriptions ──────────────────────────────────────────────────────
    depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/camera/camera/aligned_depth_to_color/image_raw", 10,
        std::bind(&TrackerNode::depthCb, this, _1));

    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        "/camera/camera/aligned_depth_to_color/camera_info", 10,
        std::bind(&TrackerNode::infoCb, this, _1));

    det_sub_ = create_subscription<vision_msgs::msg::Detection2DArray>(
        "/detector/detections", 10,
        std::bind(&TrackerNode::detectionCb, this, _1));

    // ── Publishers ─────────────────────────────────────────────────────────
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        "/tracker/markers", 10);
    obstacle_pub_ = create_publisher<overtake_msgs::msg::TrackedObstacleArray>(
        obstacle_topic_, 10);

    RCLCPP_INFO(
        get_logger(),
        "TrackerNode initialized | sensor_frame(fallback)=%s global_frame=%s topic=%s",
        sensor_frame_.c_str(), global_frame_.c_str(), obstacle_topic_.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
void TrackerNode::infoCb(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg)
{
    intrinsics_ = Intrinsics{
        msg->k[0],  // fx
        msg->k[4],  // fy
        msg->k[2],  // cx
        msg->k[5]   // cy
    };
    camera_info_frame_id_ = msg->header.frame_id;
}

// ─────────────────────────────────────────────────────────────────────────────
void TrackerNode::depthCb(
    const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
    
    cv::Mat depth;
    try {
        depth = cv_bridge::toCvCopy(msg, msg->encoding)->image;
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_WARN(get_logger(), "cv_bridge: %s", e.what());
        return;
    }

    // Ensure the mat is 16-bit unsigned
    if (depth.type() == CV_32FC1) {
        depth.convertTo(depth, CV_16UC1, 1000.0);  // m → mm
    }

    std::lock_guard<std::mutex> lock(depth_mutex_);
    depth_image_ = std::move(depth);
    depth_frame_id_ = msg->header.frame_id;
    depth_stamp_ = msg->header.stamp;
}

// ─────────────────────────────────────────────────────────────────────────────
void TrackerNode::detectionCb(
    const vision_msgs::msg::Detection2DArray::ConstSharedPtr& msg)
{

    // ── Guard: need both intrinsics and a depth image ─────────────────────
    if (!intrinsics_) {
        RCLCPP_WARN(get_logger(), "detectionCb: no intrinsics yet, dropping frame");
        return;
    }

    cv::Mat depth;
    std::string measurement_frame;
    rclcpp::Time transform_stamp = this->get_clock()->now();
    builtin_interfaces::msg::Time transform_stamp_msg = msg->header.stamp;
    {
        std::lock_guard<std::mutex> lock(depth_mutex_);
        if (depth_image_.empty()) {
            RCLCPP_WARN(get_logger(), "detectionCb: depth image empty, dropping frame");
            return;
        }
        depth = depth_image_;    // shallow copy (shared data); read-only below
        if (!depth_frame_id_.empty()) {
            measurement_frame = depth_frame_id_;
        } else if (!camera_info_frame_id_.empty()) {
            measurement_frame = camera_info_frame_id_;
        } else {
            measurement_frame = sensor_frame_;
        }
        if (depth_stamp_) {
            transform_stamp = rclcpp::Time(*depth_stamp_);
            transform_stamp_msg = *depth_stamp_;
        } else if (msg->header.stamp.sec != 0 || msg->header.stamp.nanosec != 0) {
            transform_stamp = rclcpp::Time(msg->header.stamp);
        }
    }

    if (measurement_frame.empty()) {
        RCLCPP_WARN(get_logger(), "detectionCb: no measurement frame available, dropping frame");
        return;
    }

    if (!msg->header.frame_id.empty() && msg->header.frame_id != measurement_frame) {
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Detection header frame (%s) != depth/intrinsics frame (%s). "
            "Using depth-based frame for 3D projection.",
            msg->header.frame_id.c_str(), measurement_frame.c_str());
    }

 
    const auto [fx, fy, cx0, cy0] = *intrinsics_;
    const int img_rows = depth.rows;
    const int img_cols = depth.cols;

    std::vector<Eigen::Vector3d> points_sensor;
    points_sensor.reserve(msg->detections.size());

    for (size_t det_idx = 0; det_idx < msg->detections.size(); ++det_idx) {
        const auto& det = msg->detections[det_idx];
        const int u = static_cast<int>(det.bbox.center.position.x);
        const int v = static_cast<int>(det.bbox.center.position.y);

        // 5×5 patch around (u, v) — clamped to image bounds
        const int x1 = std::max(u - 2, 0);
        const int x2 = std::min(u + 3, img_cols);
        const int y1 = std::max(v - 2, 0);
        const int y2 = std::min(v + 3, img_rows);

        if (x1 >= x2 || y1 >= y2) {
            continue;
        }

        // Collect valid (> 0) depth values from patch
        const cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
        cv::Mat patch;
        depth(roi).convertTo(patch, CV_32F);

        std::vector<float> valid_depths;
        valid_depths.reserve(patch.total());
        for (int r = 0; r < patch.rows; ++r) {
            const float* row = patch.ptr<float>(r);
            for (int c = 0; c < patch.cols; ++c)
                if (row[c] > 0.f)
                    valid_depths.push_back(row[c]);
        }

        if (valid_depths.empty()) {
            continue;
        }

        // Median depth (mm → m)
        const std::size_t mid = valid_depths.size() / 2;
        std::nth_element(valid_depths.begin(),
                         valid_depths.begin() + static_cast<std::ptrdiff_t>(mid),
                         valid_depths.end());
        const double z = static_cast<double>(valid_depths[mid]) * 0.001;

        const double x = (u - cx0) * z / fx;
        const double y = (v - cy0) * z / fy;

        // Depth deprojection is naturally expressed in the camera optical
        // frame (x right, y down, z forward). Only rotate into camera_link-style
        // axes if we are explicitly falling back to a non-optical source frame.
        if (isOpticalFrame(measurement_frame)) {
            points_sensor.emplace_back(x, y, z);
        } else {
            points_sensor.emplace_back(z, -x, -y);
        }
    }

    std::vector<Eigen::Vector3d> points_global;
    points_global.reserve(points_sensor.size());

    if (!points_sensor.empty()) {
        geometry_msgs::msg::TransformStamped tf_global_measurement;
        try {
            tf_global_measurement = tf_buffer_->lookupTransform(
                global_frame_, measurement_frame, transform_stamp);
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 1000,
                "TF %s->%s lookup failed: %s. Dropping %zu measurements for this frame.",
                measurement_frame.c_str(), global_frame_.c_str(), ex.what(),
                points_sensor.size());
            points_sensor.clear();
        }

        for (const auto& point_sensor : points_sensor) {
            geometry_msgs::msg::PointStamped point_msg_sensor;
            point_msg_sensor.header.frame_id = measurement_frame;
            point_msg_sensor.header.stamp = transform_stamp_msg;
            point_msg_sensor.point.x = point_sensor.x();
            point_msg_sensor.point.y = point_sensor.y();
            point_msg_sensor.point.z = point_sensor.z();

            geometry_msgs::msg::PointStamped point_msg_global;
            tf2::doTransform(point_msg_sensor, point_msg_global, tf_global_measurement);
            points_global.emplace_back(
                point_msg_global.point.x,
                point_msg_global.point.y,
                point_msg_global.point.z);
        }
    }

    // ── Tracker step ──────────────────────────────────────────────────────
    auto result = tracker_.step(points_global);
    publishMarkers(result.obstacles, global_frame_);
    publishDeleteMarkers(result.dead_ids, global_frame_);
    publishObstacles(result.obstacles, result.dead_ids, msg->header.stamp);
}

// ─────────────────────────────────────────────────────────────────────────────
void TrackerNode::publishMarkers(
    const std::vector<Obstacle>& obstacles,
    const std::string& frame_id)
{
    visualization_msgs::msg::MarkerArray array;
    auto now = get_clock()->now();

    for (const auto& ob : obstacles)
    {
        // ─────────────────────────────────────────────
        // LINE STRIP (trajectory history)
        // ─────────────────────────────────────────────
        visualization_msgs::msg::Marker m;
        m.header.frame_id = frame_id;
        m.header.stamp = now;

        m.ns = "tracks";
        m.id = ob.id;
        m.type = visualization_msgs::msg::Marker::LINE_STRIP;
        m.action = visualization_msgs::msg::Marker::ADD;

        m.scale.x = 0.05;

        std::mt19937 gen(ob.id);
        std::uniform_real_distribution<float> dist(0.2f, 1.0f);

        m.color.r = dist(gen);
        m.color.g = dist(gen);
        m.color.b = dist(gen);
        m.color.a = 1.0f;

        m.points.clear();
        for (const auto& p : ob.history)
        {
            geometry_msgs::msg::Point pt;
            pt.x = p.x();
            pt.y = p.y();
            pt.z = p.z();
            m.points.push_back(pt);
        }

        array.markers.push_back(m);

        // ─────────────────────────────────────────────
        // CURRENT POSITION (highlight marker)
        // ─────────────────────────────────────────────
        if (!ob.history.empty())
        {
            const auto& p = ob.history.back();

            visualization_msgs::msg::Marker curr;
            curr.header.frame_id = frame_id;
            curr.header.stamp = now;

            curr.ns = "tracks_current";
            curr.id = ob.id;

            curr.type = visualization_msgs::msg::Marker::SPHERE;
            curr.action = visualization_msgs::msg::Marker::ADD;

            curr.pose.position.x = p.x();
            curr.pose.position.y = p.y();
            curr.pose.position.z = p.z();
            curr.pose.orientation.w = 1.0;

            curr.scale.x = 0.12;
            curr.scale.y = 0.12;
            curr.scale.z = 0.12;

            curr.color.r = 1.0f;
            curr.color.g = 0.2f;
            curr.color.b = 0.2f;
            curr.color.a = 1.0;

            array.markers.push_back(curr);
        }
        if (!ob.predicted_trajectory.empty())
        {
            visualization_msgs::msg::Marker pred;
            pred.header.frame_id = frame_id;
            pred.header.stamp    = now;

            pred.ns     = "tracks_predicted";  // separate namespace
            pred.id     = ob.id;
            pred.type   = visualization_msgs::msg::Marker::LINE_STRIP;
            pred.action = visualization_msgs::msg::Marker::ADD;

            pred.scale.x = 0.03;  // thinner than history

            // Same color as track but faded
            pred.color.r = m.color.r;
            pred.color.g = m.color.g;
            pred.color.b = m.color.b;
            pred.color.a = 0.4f;  // transparent

            for (const auto& p : ob.predicted_trajectory)
            {
                geometry_msgs::msg::Point pt;
                pt.x = p.position.x();
                pt.y = p.position.y();
                pt.z = p.position.z();
                pred.points.push_back(pt);
            }

            array.markers.push_back(pred);
        }
    }

    marker_pub_->publish(array);
}

void TrackerNode::publishObstacles(
    const std::vector<Obstacle>& obstacles,
    const std::vector<int>& dead_ids,
    const builtin_interfaces::msg::Time& stamp)
{
    overtake_msgs::msg::TrackedObstacleArray arr;
    arr.header.frame_id = global_frame_;
    arr.header.stamp = stamp;

    arr.dead_ids.reserve(dead_ids.size());
    for (int id : dead_ids) {
        arr.dead_ids.push_back(static_cast<int32_t>(id));
    }

    arr.obstacles.reserve(obstacles.size());
    for (const auto& ob : obstacles) {
        // Class-based radius. Detector is currently person-only, so we map
        // Unknown -> person default here. Tune per-class when the detector
        // emits more classes.
        float radius = 0.22f;
        switch (ob.c) {
            case ObstacleClass::Ball: radius = 0.12f; break;
            case ObstacleClass::Cone: radius = 0.18f; break;
            case ObstacleClass::Person:
            case ObstacleClass::Unknown:
            default:                  radius = 0.22f; break;
        }

        overtake_msgs::msg::TrackedObstacle tracked;
        tracked.id = static_cast<int32_t>(ob.id);
        tracked.cls = static_cast<int32_t>(ob.c);
        tracked.pose.position.x = ob.position.x();
        tracked.pose.position.y = ob.position.y();
        tracked.pose.position.z = ob.position.z();
        if (ob.velocity.squaredNorm() > 1e-6) {
            tracked.pose.orientation = yawToQuaternion(ob.yaw);
        } else {
            tracked.pose.orientation.w = 1.0;
        }
        tracked.velocity.x = ob.velocity.x();
        tracked.velocity.y = ob.velocity.y();
        tracked.velocity.z = 0.0;
        tracked.radius = radius;
        arr.obstacles.push_back(tracked);
    }

    obstacle_pub_->publish(arr);
}

void TrackerNode::publishDeleteMarkers(
    const std::vector<int>& dead_ids,
    const std::string& frame_id)
{
    if (dead_ids.empty()) return;

    visualization_msgs::msg::MarkerArray array;
    auto now = get_clock()->now();

    for (int id : dead_ids)
    {
        // Delete the line strip
        visualization_msgs::msg::Marker m;
        m.header.stamp = now;
        m.header.frame_id = frame_id;
        m.ns = "tracks";
        m.id = id;
        m.action = visualization_msgs::msg::Marker::DELETE;
        array.markers.push_back(m);

        // Delete the sphere
        visualization_msgs::msg::Marker curr;
        curr.header.stamp = now;
        curr.header.frame_id = frame_id;
        curr.ns = "tracks_current";   // ← must match the ns used in publishMarkers
        curr.id = id;
        curr.action = visualization_msgs::msg::Marker::DELETE;
        array.markers.push_back(curr);

        visualization_msgs::msg::Marker pred;
        pred.header.stamp    = now;
        pred.header.frame_id = frame_id;
        pred.ns     = "tracks_predicted";  // ← match above
        pred.id     = id;
        pred.action = visualization_msgs::msg::Marker::DELETE;
        array.markers.push_back(pred);
    }

    marker_pub_->publish(array);
}
