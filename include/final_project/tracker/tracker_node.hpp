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
#include <overtake_msgs/msg/tracked_obstacle_array.hpp>

#include <message_filters/subscriber.h>                       // CHANGED
#include <message_filters/synchronizer.h>                    // CHANGED
#include <message_filters/sync_policies/approximate_time.h>  // CHANGED

#include "final_project/tracker/track_manager.hpp"

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace final_project::tracker {

struct Intrinsics {
    double fx = 608.489501953125;
    double fy = 608.42236328125;
    double cx = 312.7839050292969;
    double cy = 245.48492431640625;
};

class TrackerNode : public rclcpp::Node {
public:
    TrackerNode();

private:
    //one synchronized callback for detections + LaserScan.
    void fusedCb(const vision_msgs::msg::Detection2DArray::ConstSharedPtr& det_msg,
                 const sensor_msgs::msg::LaserScan::ConstSharedPtr& scan_msg);

    void publishMarkers(const std::vector<Obstacle>& obstacles,
                        const std::string& frame_id);
    void publishDeleteMarkers(const std::vector<int>& dead_ids,
                              const std::string& frame_id);
    void publishObstacles(const Step& step,
                          const rclcpp::Time& stamp,
                          const std::string& frame_id);

    // ApproximateTime sync policy for camera detections and LiDAR.
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        vision_msgs::msg::Detection2DArray,
        sensor_msgs::msg::LaserScan>;

    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;

    message_filters::Subscriber<vision_msgs::msg::Detection2DArray> det_sub_;
    message_filters::Subscriber<sensor_msgs::msg::LaserScan> scan_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<overtake_msgs::msg::TrackedObstacleArray>::SharedPtr obs_pub_;

    Intrinsics intrinsics_;
    std::optional<rclcpp::Time> last_update_stamp_;

    std::string camera_frame_ = "camera_link";
    std::string map_frame_ = "map";
    std::string scan_frame_ = "laser";
    const int img_cols_ = 640;
    const int img_rows_ = 480;

    // Must be >= MPC overtake activation_lookahead (~7.0 m) so the planner
    // sees obstacles before it has to react.
    const double max_distance_for_detection_ = 8.0;


    TrackManager2D tracker_;
    tf2_ros::Buffer tf_buffer_{get_clock()};
    tf2_ros::TransformListener tf_listener_{tf_buffer_};

};

}  // namespace final_project::tracker