#include "final_project/tracker/tracker_node.hpp"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TrackerNode>());
    rclcpp::shutdown();
    return 0;
}