#pragma once

#include <deque>
#include <Eigen/Dense>
#include "final_project/tracker/kalman_filter_3d.hpp"
#include <memory>

struct TrajectoryHistory{
    int id = -1;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    std::deque<Eigen::Vector3d> history{};
};

struct StepResult {
    std::vector<TrajectoryHistory> histories;
    std::vector<int> dead_ids;
};

struct Trajectory {
    std::unique_ptr<KalmanFilter3D> kf;
    int age = 0;
    int missed = 0;
    TrajectoryHistory r;
};

class MultiTracker3D {
public:
    explicit MultiTracker3D(double dt = 1.0 / 30.0);

    StepResult step(const std::vector<Eigen::Vector3d>& measurements);


private:
    void spawn(const Eigen::Vector3d& measurement);

    std::vector<Trajectory> traj_;
    int next_id_ = 0;

    double gating_threshold_ = 7.81; // Mahalanobis distance threshold
    int max_missed_ = 15;
    double dt_;
};
