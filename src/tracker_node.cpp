#include "final_project/tracker/tracker_node.hpp"

#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <opencv2/imgproc.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <random>


using std::placeholders::_1;

// ─────────────────────────────────────────────────────────────────────────────
TrackerNode::TrackerNode()
    : Node("tracker_node"),
      tracker_(1.0 / 30.0)
{
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

    

    // ── Publisher ──────────────────────────────────────────────────────────
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
    "/tracker/markers", 10);

    RCLCPP_INFO(get_logger(), "TrackerNode initialized");
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
    {
        std::lock_guard<std::mutex> lock(depth_mutex_);
        if (depth_image_.empty()) {
            RCLCPP_WARN(get_logger(), "detectionCb: depth image empty, dropping frame");
            return;
        }
        depth = depth_image_;    // shallow copy (shared data); read-only below
    }

 
    const auto [fx, fy, cx0, cy0] = *intrinsics_;
    const int img_rows = depth.rows;
    const int img_cols = depth.cols;

    std::vector<Eigen::Vector3d> points_3d;
    points_3d.reserve(msg->detections.size());

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

        // Python code uses [z, x, -y] ordering
        points_3d.emplace_back(z, x, -y);
    }

    // ── Tracker step ──────────────────────────────────────────────────────
    auto result = tracker_.step(points_3d);
    publishMarkers(result.obstacles);
    publishDeleteMarkers(result.dead_ids);
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