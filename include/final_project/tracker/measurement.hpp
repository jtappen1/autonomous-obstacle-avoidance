#pragma once

#include <Eigen/Dense>

namespace final_project::tracker {

enum class ObstacleClass : int {
    Unknown = -1,
    Person = 0,
    Ball = 1,
    Cone = 2,
};

struct DetectionMeasurement {
    Eigen::Vector2d pos = Eigen::Vector2d::Zero();   // [forward, lateral]
    Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * 0.1;
    double height = 0.0;                             // vertical channel, not filtered
    double confidence = 0.0;
    ObstacleClass obstacle_class = ObstacleClass::Unknown;
};

}  // namespace final_project::tracker