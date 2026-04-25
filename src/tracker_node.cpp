#include "final_project/tracker/tracker_node.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <tf2/LinearMath/Matrix3x3.h>   // CHANGED
#include <tf2/LinearMath/Transform.h>   // CHANGED
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace final_project::tracker {

using std::placeholders::_1;
using std::placeholders::_2;  // CHANGED

namespace {

// CHANGED: LiDAR point projected into the image.
struct ProjectedLidarPoint {
    int scan_index = -1;
    double range = 0.0;
    double u = 0.0;
    double v = 0.0;
    Eigen::Vector2d xy_scan = Eigen::Vector2d::Zero();  // 2D point in scan frame
};

// CHANGED: candidate cluster inside one bbox.
struct ClusterCandidate {
    int begin = 0;
    int end = 0;
    int count = 0;

    double median_range = std::numeric_limits<double>::infinity();
    double median_u = 0.0;
    double median_v = 0.0;
    double score = std::numeric_limits<double>::infinity();

    Eigen::Vector2d centroid_scan = Eigen::Vector2d::Zero();
    Eigen::Matrix2d cov_scan = Eigen::Matrix2d::Identity();
};

double medianInPlace(std::vector<double>& values) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(),
                     values.begin() + static_cast<std::ptrdiff_t>(mid),
                     values.end());
    return values[mid];
}

double adaptiveGapThreshold(double range_m) {
    // CHANGED: adaptive Euclidean split threshold.
    return 0.08 + 0.15 * range_m;
}

bool shouldSplitCluster(const ProjectedLidarPoint& prev,
                        const ProjectedLidarPoint& curr) {
    
    // Calculate the actual physical distance between the two laser hits
    const double gap = (curr.xy_scan - prev.xy_scan).norm();
    
    // Dynamic threshold: allows larger gaps for points further away
    const double thresh =
        std::max(adaptiveGapThreshold(prev.range), adaptiveGapThreshold(curr.range));
        
    // Break the cluster ONLY if the physical gap is unreasonably large
    return gap > thresh;
}

Eigen::Vector2d componentwiseMedian(const std::vector<ProjectedLidarPoint>& pts,
                                    int begin,
                                    int end) {
    std::vector<double> xs;
    std::vector<double> ys;
    xs.reserve(std::max(0, end - begin));
    ys.reserve(std::max(0, end - begin));

    for (int i = begin; i < end; ++i) {
        xs.push_back(pts[i].xy_scan.x());
        ys.push_back(pts[i].xy_scan.y());
    }

    return Eigen::Vector2d(medianInPlace(xs), medianInPlace(ys));
}

Eigen::Matrix2d covarianceAroundCentroid(const std::vector<ProjectedLidarPoint>& pts,
                                         int begin,
                                         int end,
                                         const Eigen::Vector2d& centroid,
                                         double median_range) {
    Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
    const int count = end - begin;

    if (count >= 2) {
        for (int i = begin; i < end; ++i) {
            const Eigen::Vector2d d = pts[i].xy_scan - centroid;
            cov += d * d.transpose();
        }
        cov /= static_cast<double>(count - 1);
    }

    // CHANGED: covariance floor grows slightly with range and small cluster size.
    const double sigma_floor = 0.04 + 0.01 * median_range + (count < 2 ? 0.15 : 0.0);
    cov += sigma_floor * sigma_floor * Eigen::Matrix2d::Identity();

    cov = 0.5 * (cov + cov.transpose());
    return cov;
}

ClusterCandidate buildClusterCandidate(
    const std::vector<ProjectedLidarPoint>& pts,
    int begin,
    int end,
    double bbox_u,
    double bbox_v,
    double bbox_w,
    double bbox_h) 
{
    ClusterCandidate c;
    c.begin = begin;
    c.end = end;
    c.count = end - begin;

    if (c.count <= 0) return c;

    std::vector<double> ranges;
    std::vector<double> us;
    std::vector<double> vs;
    ranges.reserve(c.count);
    us.reserve(c.count);
    vs.reserve(c.count);

    for (int i = begin; i < end; ++i) {
        ranges.push_back(pts[i].range);
        us.push_back(pts[i].u);
        vs.push_back(pts[i].v);
    }

    c.median_range = medianInPlace(ranges);
    c.median_u = medianInPlace(us);
    c.median_v = medianInPlace(vs);
    c.centroid_scan = componentwiseMedian(pts, begin, end);
    c.cov_scan = covarianceAroundCentroid(pts, begin, end, c.centroid_scan, c.median_range);

    // CHANGED:
    // Use closeness to bbox center as a CLUSTER score, not direct point weights for centroid.
    // const double half_w = std::max(0.5 * bbox_w, 1.0);
    // const double half_h = std::max(0.5 * bbox_h, 1.0);
    // const double du = (c.median_u - bbox_u) / half_w;
    // const double dv = (c.median_v - bbox_v) / half_h;
    // const double center_penalty = du * du + dv * dv;

    const double half_w = std::max(0.5 * bbox_w, 1.0);
    const double du = (c.median_u - bbox_u) / half_w;
    const double center_penalty = du * du;

    // CHANGED: prefer front-most plausible cluster, but reward support count.
    const double count_bonus = 0.1 * std::min(c.count, 20);
    c.score = c.median_range + 0.5 * center_penalty - count_bonus;

    return c;
}

std::optional<ClusterCandidate> selectBestCluster(const std::vector<ProjectedLidarPoint>& pts,
                                                  double bbox_u,
                                                  double bbox_v,
                                                  double bbox_w,
                                                  double bbox_h) {
    if (pts.empty()) return std::nullopt;

    std::vector<ClusterCandidate> clusters;
    int start = 0;

    for (int i = 1; i < static_cast<int>(pts.size()); ++i) {
        if (shouldSplitCluster(pts[i - 1], pts[i])) {
            clusters.push_back(buildClusterCandidate(pts, start, i, bbox_u, bbox_v, bbox_w, bbox_h));
            start = i;
        }
    }
    clusters.push_back(buildClusterCandidate(
        pts, start, static_cast<int>(pts.size()), bbox_u, bbox_v, bbox_w, bbox_h));

    auto best_it = std::min_element(
        clusters.begin(), clusters.end(),
        [](const ClusterCandidate& a, const ClusterCandidate& b) {
            return a.score < b.score;
        });

    if (best_it == clusters.end() || best_it->count <= 0) return std::nullopt;
    return *best_it;
}

std::string resolveMarkerFrame(const std::string& header_frame_id,
                               const std::string& fallback) {
    return header_frame_id.empty() ? fallback : header_frame_id;
}

}  // namespace

TrackerNode::TrackerNode()
    : Node("tracker_node"),
      tracker_(1.0 / 30.0) {
    // CHANGED: keep CameraInfo subscription for intrinsics + optical frame.
    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        "/camera/camera/aligned_depth_to_color/camera_info",
        rclcpp::SensorDataQoS(),
        std::bind(&TrackerNode::infoCb, this, _1));

    //Synchronized detection + LiDAR subscriptions.
    det_sub_.subscribe(this, "/detector/detections");
    scan_sub_.subscribe(this, "/scan");
    
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(10), det_sub_, scan_sub_);

    // Approximate sync window, tune between 30–60 ms.
    sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.05));
    sync_->registerCallback(std::bind(&TrackerNode::fusedCb, this, _1, _2));

    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        "/tracker/markers", 10);

    RCLCPP_INFO(get_logger(), "TrackerNode initialized (LiDAR + detections)");
}

void TrackerNode::infoCb(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg) {
    intrinsics_ = Intrinsics{msg->k[0], msg->k[4], msg->k[2], msg->k[5]};
}

void TrackerNode::fusedCb(
    const vision_msgs::msg::Detection2DArray::ConstSharedPtr& det_msg,
    const sensor_msgs::msg::LaserScan::ConstSharedPtr& scan_msg) {
    if (!intrinsics_) {
        RCLCPP_WARN(get_logger(), "fusedCb: no intrinsics yet, dropping frame");
        return;
    }
    const auto [fx, fy, cx0, cy0] = *intrinsics_;
    const int img_cols = std::max(1, static_cast<int>(std::round(2.0 * cx0)));
    const int img_rows = std::max(1, static_cast<int>(std::round(2.0 * cy0)));

    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
    "IMAGE SIZE (estimated): cols=%d rows=%d  cx0=%.1f cy0=%.1f",
    img_cols, img_rows, cx0, cy0);

    const rclcpp::Time stamp =
        (det_msg->header.stamp.sec == 0 && det_msg->header.stamp.nanosec == 0)
            ? rclcpp::Time(scan_msg->header.stamp)
            : rclcpp::Time(det_msg->header.stamp);

    geometry_msgs::msg::TransformStamped tf_scan_to_camera_msg;
    geometry_msgs::msg::TransformStamped tf_scan_to_track_msg;

    try {
        tf_scan_to_camera_msg = tf_buffer_.lookupTransform(
            camera_frame_, scan_msg->header.frame_id, tf2::TimePointZero);

        tf_scan_to_track_msg = tf_buffer_.lookupTransform(
            tracking_frame_, scan_msg->header.frame_id, stamp);
    } catch (const tf2::TransformException& e) {
        RCLCPP_WARN(get_logger(), "TF lookup failed: %s", e.what());
        return;
    }

    tf2::Transform tf_scan_to_camera;
    tf2::Transform tf_scan_to_track;
    tf2::fromMsg(tf_scan_to_camera_msg.transform, tf_scan_to_camera);
    tf2::fromMsg(tf_scan_to_track_msg.transform, tf_scan_to_track);

    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
    "TF: scan_frame='%s'  camera_frame='%s'  tracking_frame='%s'",
    scan_msg->header.frame_id.c_str(),
    camera_frame_.c_str(),
    tracking_frame_.c_str());

    const tf2::Vector3 origin = tf_scan_to_camera.getOrigin();
    const tf2::Matrix3x3 rot(tf_scan_to_camera.getRotation());
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "TF scan->camera: translation=(%.3f, %.3f, %.3f)",
        origin.x(), origin.y(), origin.z());
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "TF scan->camera rotation matrix:\n"
        "  [%.3f  %.3f  %.3f]\n"
        "  [%.3f  %.3f  %.3f]\n"
        "  [%.3f  %.3f  %.3f]",
        rot[0][0], rot[0][1], rot[0][2],
        rot[1][0], rot[1][1], rot[1][2],
        rot[2][0], rot[2][1], rot[2][2]);

    // CHANGED: 2D rotation to map/odom for covariance rotation.
    const tf2::Matrix3x3 R3(tf_scan_to_track.getRotation());
    Eigen::Matrix2d R_track_scan;
    R_track_scan << R3[0][0], R3[0][1],
                    R3[1][0], R3[1][1];


    const double FOV_LIMIT = 1.396; // ~80 degrees
    
    int start_idx = static_cast<int>(std::round((-FOV_LIMIT - scan_msg->angle_min) / scan_msg->angle_increment));
    int end_idx   = static_cast<int>(std::round((FOV_LIMIT - scan_msg->angle_min) / scan_msg->angle_increment));
    
    start_idx = std::max(0, start_idx);
    end_idx   = std::min(static_cast<int>(scan_msg->ranges.size() - 1), end_idx);

    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
    "SCAN: %zu ranges, angle_min=%.3f angle_max=%.3f increment=%.5f  FOV indices [%d, %d]",
    scan_msg->ranges.size(),
    scan_msg->angle_min, scan_msg->angle_max, scan_msg->angle_increment,
    start_idx, end_idx);


    // std::vector<ProjectedLidarPoint> projected_points;
    // projected_points.reserve(end_idx - start_idx + 1); // Use your expected_rays for efficiency

    // // --- Modify the loop to only check the limited indices ---
    // for (int i = start_idx; i <= end_idx; ++i) {
    //     const double range = scan_msg->ranges[i];
        
    //     if (!std::isfinite(range) ||
    //         range < scan_msg->range_min ||
    //         range > scan_msg->range_max) {
    //         continue;
    //     }

    //     const double theta =
    //         scan_msg->angle_min + static_cast<double>(i) * scan_msg->angle_increment;

    //     // ... rest of the transform and projection logic remains the same ...

    //     const double x_scan = range * std::cos(theta);
    //     const double y_scan = range * std::sin(theta);

    //     const tf2::Vector3 p_scan(x_scan, y_scan, 0.0);
    //     const tf2::Vector3 p_cam = tf_scan_to_camera * p_scan;

    //     // CHANGED:
    //     // Assumes camera optical frame convention: X right, Y down, Z forward.
    //     const double xc = p_cam.x();
    //     const double yc = p_cam.y();
    //     const double zc = p_cam.z();
    //     if (zc <= 0.05) continue;

    //     const double u = fx * (xc / zc) + cx0;
    //     const double v = fy * (yc / zc) + cy0;

    //     if (u < 0.0 || u >= img_cols || v < 0.0 || v >= img_rows) continue;

    //     ProjectedLidarPoint point;
    //     point.scan_index = i;
    //     point.range = range;
    //     point.u = u;
    //     point.v = v;
    //     point.xy_scan = Eigen::Vector2d(x_scan, y_scan);
    //     projected_points.push_back(point);
    // }
    int dbg_invalid_range = 0;
    int dbg_behind_camera = 0;
    int dbg_out_of_image  = 0;

    std::vector<ProjectedLidarPoint> projected_points;
    projected_points.reserve(end_idx - start_idx + 1);

    for (int i = start_idx; i <= end_idx; ++i) {
        const double range = scan_msg->ranges[i];

        if (!std::isfinite(range) ||
            range < scan_msg->range_min ||
            range > scan_msg->range_max) {
            ++dbg_invalid_range;
            continue;
        }

        const double theta =
            scan_msg->angle_min + static_cast<double>(i) * scan_msg->angle_increment;
        const double x_scan = range * std::cos(theta);
        const double y_scan = range * std::sin(theta);

        const tf2::Vector3 p_scan(x_scan, y_scan, 0.0);
        const tf2::Vector3 p_cam = tf_scan_to_camera * p_scan;

        const double xc = p_cam.x();
        const double yc = p_cam.y();
        const double zc = p_cam.z();

        if (zc <= 0.05) {
            ++dbg_behind_camera;
            continue;
        }

        const double u = fx * (xc / zc) + cx0;
        const double v = fy * (yc / zc) + cy0;

        if (u < 0.0 || u >= img_cols || v < 0.0 || v >= img_rows) {
            ++dbg_out_of_image;
            continue;
        }

        ProjectedLidarPoint point;
        point.scan_index = i;
        point.range      = range;
        point.u          = u;
        point.v          = v;
        point.xy_scan    = Eigen::Vector2d(x_scan, y_scan);
        projected_points.push_back(point);
    }
    // END NEW

    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
        "PROJECTION: total_in_FOV=%d  projected_ok=%zu  "
        "dropped[invalid_range=%d  behind_camera=%d  out_of_image=%d]",
        (end_idx - start_idx + 1),
        projected_points.size(),
        dbg_invalid_range, dbg_behind_camera, dbg_out_of_image);

    // Log a few sample projected points so we can see where they land
    if (!projected_points.empty()) {
        const auto& p0 = projected_points[projected_points.size() / 2];
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
            "PROJECTION sample (mid-point): range=%.2f  u=%.1f  v=%.1f  "
            "x_scan=%.2f  y_scan=%.2f",
            p0.range, p0.u, p0.v, p0.xy_scan.x(), p0.xy_scan.y());
    }

    std::vector<DetectionMeasurement> measurements;
    measurements.reserve(det_msg->detections.size());

    // CHANGED: generate one LiDAR-based measurement per detection.
    for (const auto& det : det_msg->detections) {
        const double bbox_u = det.bbox.center.position.x;
        const double bbox_v = det.bbox.center.position.y;
        const double bbox_w = det.bbox.size_x;
        const double bbox_h = det.bbox.size_y;

        // Strict bounding box limits to reduce background noise
        const double x_min = bbox_u - 0.5 * bbox_w;
        const double x_max = bbox_u + 0.5 * bbox_w;
        
        std::vector<ProjectedLidarPoint> in_bbox;
        in_bbox.reserve(32);

        for (const auto& point : projected_points) {
            if (point.u >= x_min && point.u <= x_max) {
                in_bbox.push_back(point);
            }
        }

        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
            "BBOX det[%zu]: center=(%.1f, %.1f)  size=(%.1f x %.1f)  "
            "x_range=[%.1f, %.1f]"
            "total_projected=%zu  in_bbox=%zu",
            &det - &det_msg->detections[0],
            bbox_u, bbox_v, bbox_w, bbox_h,
            x_min, x_max,
            projected_points.size(), in_bbox.size());

        if (in_bbox.empty()) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                "BBOX MISS: no projected points fell inside bbox — "
                "check if u/v ranges overlap above");
            continue;
        }

        // CHANGED: cluster points in scan order and pick best cluster.
        const auto best_cluster =
            selectBestCluster(in_bbox, bbox_u, bbox_v, bbox_w, bbox_h);
        if (!best_cluster.has_value())
        {
            RCLCPP_INFO(get_logger(), "best cluster has NO value");
            continue;

        } 

        const tf2::Vector3 centroid_scan(
            best_cluster->centroid_scan.x(),
            best_cluster->centroid_scan.y(),
            0.0);
        const tf2::Vector3 centroid_track = tf_scan_to_track * centroid_scan;

        DetectionMeasurement measurement;
        measurement.pos = Eigen::Vector2d(centroid_track.x(), centroid_track.y());
        measurement.height = 0.0;

        // CHANGED: rotate cluster covariance into tracking frame.
        Eigen::Matrix2d R_track =
            R_track_scan * best_cluster->cov_scan * R_track_scan.transpose();
        R_track = 0.5 * (R_track + R_track.transpose());
        R_track.diagonal().array() += 1e-4;
        measurement.R = R_track;

        measurements.push_back(measurement);
    }

    double dt = 1.0 / 30.0;
    if (last_update_stamp_) {
        dt = (stamp - *last_update_stamp_).seconds();
        if (!std::isfinite(dt) || dt <= 0.0 || dt > 0.25) {
            dt = 1.0 / 30.0;
        }
    }
    last_update_stamp_ = stamp;

    // --- Logging the measurements before they enter the tracker ---
    for (size_t i = 0; i < measurements.size(); ++i) {
        const auto& m = measurements[i];
        RCLCPP_INFO(
            this->get_logger(),
            "Measurement [%zu]: x = %.3f, y = %.3f, height = %.3f",
            i, m.pos.x(), m.pos.y(), m.height
        );
    }

    // Existing tracker step
    const auto result = tracker_.step(measurements, dt);
    publishMarkers(result.obstacles, tracking_frame_);
    publishDeleteMarkers(result.dead_ids, tracking_frame_);
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
            pt.z = 0.0;
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
        current.pose.position.z = 0.0;
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
            pt.z = 0.0;
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