#include "final_project/tracker/track_manager.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace final_project::tracker {

TrackManager2D::TrackManager2D(double default_dt) : default_dt_(default_dt) {}

void TrackManager2D::spawn(const DetectionMeasurement& measurement, double dt) {
    tracks_.emplace_back(next_id_++, measurement, dt, std_a_, std_yawdd_);
}

TrackManager2D::CostMatrix TrackManager2D::buildCostMatrix(
    const std::vector<DetectionMeasurement>& measurements) const {
    CostMatrix cost(tracks_.size(), std::vector<double>(measurements.size(), unassigned_cost_));

    for (std::size_t i = 0; i < tracks_.size(); ++i) {
        for (std::size_t j = 0; j < measurements.size(); ++j) {
            const auto& track = tracks_[i];
            const auto& meas = measurements[j];
            const Eigen::Vector2d r = meas.pos - track.ukf.position();
            Eigen::Matrix2d S = track.ukf.innovationCovariance(meas.R);
            const double d2 = r.transpose() * S.ldlt().solve(r);
            if (std::isfinite(d2) && d2 <= gating_threshold_) {
                cost[i][j] = d2;
            }
        }
    }
    return cost;
}

std::vector<int> TrackManager2D::solveHungarian(const CostMatrix& cost) const {
    const int n_rows = static_cast<int>(cost.size());
    const int n_cols = n_rows == 0 ? 0 : static_cast<int>(cost.front().size());
    const int n = std::max(n_rows, n_cols);
    if (n == 0) return {};

    std::vector<std::vector<double>> a(n + 1, std::vector<double>(n + 1, unassigned_cost_));
    for (int i = 1; i <= n_rows; ++i) {
        for (int j = 1; j <= n_cols; ++j) {
            a[i][j] = cost[i - 1][j - 1];
        }
    }

    std::vector<double> u(n + 1, 0.0), v(n + 1, 0.0);
    std::vector<int> p(n + 1, 0), way(n + 1, 0);

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(n + 1, std::numeric_limits<double>::infinity());
        std::vector<bool> used(n + 1, false);
        do {
            used[j0] = true;
            int i0 = p[j0];
            double delta = std::numeric_limits<double>::infinity();
            int j1 = 0;
            for (int j = 1; j <= n; ++j) {
                if (used[j]) continue;
                const double cur = a[i0][j] - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    std::vector<int> assignment(n_rows, -1);
    for (int j = 1; j <= n; ++j) {
        if (p[j] > 0 && p[j] <= n_rows && j <= n_cols) {
            assignment[p[j] - 1] = j - 1;
        }
    }
    return assignment;
}

Obstacle TrackManager2D::makeObstacle(const Track& track, double dt) const {
    Obstacle obstacle;
    obstacle.id = track.id;
    obstacle.obstacle_class = track.obstacle_class;
    obstacle.position = Eigen::Vector3d(track.ukf.position().x(), track.ukf.position().y(), track.last_height);
    obstacle.velocity = track.ukf.velocity();
    obstacle.yaw = track.ukf.yaw();
    obstacle.confidence = track.confidence;
    obstacle.history = track.history;

    const auto future = track.ukf.predictTrajectory(num_predicted_steps_, dt);
    obstacle.predicted_trajectory.reserve(future.size());
    for (const auto& f : future) {
        PredictedState pred;
        pred.position = Eigen::Vector3d(f.position.x(), f.position.y(), track.last_height);
        pred.velocity = f.velocity;
        pred.yaw = f.yaw;
        pred.covariance = f.covariance;
        obstacle.predicted_trajectory.push_back(pred);
    }
    return obstacle;
}

Step TrackManager2D::step(const std::vector<DetectionMeasurement>& measurements, double dt) {
    const double used_dt = (std::isfinite(dt) && dt > 1e-4) ? dt : default_dt_;

    for (auto& track : tracks_) {
        track.ukf.predict(used_dt);
        track.age++;
    }

    std::vector<bool> detection_used(measurements.size(), false);
    std::vector<int> dead_ids;

    if (!tracks_.empty() && !measurements.empty()) {
        const auto cost = buildCostMatrix(measurements);
        const auto assignment = solveHungarian(cost);

        for (std::size_t i = 0; i < tracks_.size(); ++i) {
            const int j = assignment[i];
            if (j >= 0 && cost[i][j] < unassigned_cost_) {
                const auto& meas = measurements[j];
                tracks_[i].ukf.updatePosition(meas.pos, meas.R);
                tracks_[i].missed = 0;
                tracks_[i].hits++;
                tracks_[i].last_height = meas.height;
                tracks_[i].confidence = meas.confidence;
                tracks_[i].obstacle_class = meas.obstacle_class;
                tracks_[i].history.push_back(Eigen::Vector3d(meas.pos.x(), meas.pos.y(), meas.height));
                if (static_cast<int>(tracks_[i].history.size()) > max_history_) {
                    tracks_[i].history.pop_front();
                }
                detection_used[j] = true;
            } else {
                tracks_[i].missed++;
            }
        }
    } else {
        for (auto& track : tracks_) {
            track.missed++;
        }
    }

    // for (std::size_t j = 0; j < measurements.size(); ++j) {
    //     if (!detection_used[j]) {
    //         spawn(measurements[j], used_dt);
    //     }
    // }
    for (std::size_t j = 0; j < measurements.size(); ++j) {
        if (!detection_used[j]) {
            // don't spawn if too close to an existing track
            bool too_close = false;
            for (const auto& track : tracks_) {
                const double dist = (measurements[j].pos - track.ukf.position()).norm();
                if (dist < min_spawn_distance_) {
                    too_close = true;
                    break;
                }
            }
            if (!too_close) {
                spawn(measurements[j], used_dt);
            }
        }
    }

    auto it = std::remove_if(tracks_.begin(), tracks_.end(), [&](const Track& track) {
        if (track.missed > max_missed_) {
            dead_ids.push_back(track.id);
            return true;
        }
        return false;
    });
    tracks_.erase(it, tracks_.end());

    Step result;
    result.dead_ids = std::move(dead_ids);
    for (const auto& track : tracks_) {
        if (track.hits >= min_confirmed_hits_) {
            result.obstacles.push_back(makeObstacle(track, used_dt));
        }
    }
    return result;
}

}  // namespace final_project::tracker