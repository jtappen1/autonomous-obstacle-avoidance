#pragma once

#include <deque>
#include <Eigen/Dense>
#include "final_project/tracker/kalman_filter.hpp"
#include <memory>
#include <vector>

enum class ObstacleClass : int {
    Unknown = -1,
    Person = 0,
    Ball = 1,
    Cone = 2,
};

struct Obstacle {
    int id = -1;
    ObstacleClass c = ObstacleClass::Unknown;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector2d velocity = Eigen::Vector2d::Zero();
    double yaw = 0.0;
    std::vector<PredictedState> predicted_trajectory;
    std::deque<Eigen::Vector3d> history{};
};

struct Step {
    std::vector<Obstacle> obstacles;
    std::vector<int> dead_ids;
};

struct Filter {
    std::unique_ptr<KalmanFilter3D> kf;
    int age = 0;
    int missed = 0;
    Obstacle obs;
};

class MultiTracker3D {
public:
    explicit MultiTracker3D(double dt = 1.0 / 30.0);

    Step step(const std::vector<Eigen::Vector3d>& measurements);


private:
    void spawn(const Eigen::Vector3d& measurement);

    std::vector<Filter> filter_;
    int next_id_ = 0;

    double gating_threshold_ = 4.61; // Mahalanobis distance threshold
    int min_age_ = 60;
    int max_missed_ = 30;
    double dt_;
    int num_predicted_steps_ = 30;
};
