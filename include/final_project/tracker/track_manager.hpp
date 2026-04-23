#pragma once

#include <deque>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "final_project/tracker/measurement.hpp"
#include "final_project/tracker/UKF.hpp"

namespace final_project::tracker {

struct PredictedState {
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector2d velocity = Eigen::Vector2d::Zero();
    double yaw = 0.0;
    Eigen::Matrix<double, 5, 5> covariance = Eigen::Matrix<double, 5, 5>::Zero();
};

struct Obstacle {
    int id = -1;
    ObstacleClass obstacle_class = ObstacleClass::Unknown;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector2d velocity = Eigen::Vector2d::Zero();
    double yaw = 0.0;
    double confidence = 0.0;
    std::vector<PredictedState> predicted_trajectory;
    std::deque<Eigen::Vector3d> history;
};

struct Step {
    std::vector<Obstacle> obstacles;
    std::vector<int> dead_ids;
};

class TrackManager2D {
public:
    explicit TrackManager2D(double default_dt = 1.0 / 30.0);

    Step step(const std::vector<DetectionMeasurement>& measurements, double dt);

private:
    struct Track {
        Track(int track_id,
              const DetectionMeasurement& measurement,
              double dt,
              double std_a,
              double std_yawdd)
            : id(track_id),
              ukf(measurement.pos, measurement.R, dt, std_a, std_yawdd),
              last_height(measurement.height),
              confidence(measurement.confidence),
              obstacle_class(measurement.obstacle_class) {
            history.push_back(Eigen::Vector3d(measurement.pos.x(), measurement.pos.y(), measurement.height));
        }

        int id = -1;
        UKF_CTRV ukf;
        int age = 1;
        int hits = 1;
        int missed = 0;
        double last_height = 0.0;
        double confidence = 0.0;
        ObstacleClass obstacle_class = ObstacleClass::Unknown;
        std::deque<Eigen::Vector3d> history;
    };

    using CostMatrix = std::vector<std::vector<double>>;

    void spawn(const DetectionMeasurement& measurement, double dt);
    CostMatrix buildCostMatrix(const std::vector<DetectionMeasurement>& measurements) const;
    std::vector<int> solveHungarian(const CostMatrix& cost) const;
    Obstacle makeObstacle(const Track& track, double dt) const;

    std::vector<Track> tracks_;
    int next_id_ = 0;

    double default_dt_ = 1.0 / 30.0;
    double std_a_ = 2.0;
    double std_yawdd_ = 1.0;
    double gating_threshold_ = 5.991;   // chi-square 95% for 2 DoF
    double unassigned_cost_ = 1e6;
    int min_confirmed_hits_ = 3;
    int max_missed_ = 10;
    int max_history_ = 60;
    int num_predicted_steps_ = 30;
};

}  // namespace final_project::tracker