#include "final_project/tracker/multi_tracker.hpp"
#include <memory>
#include <algorithm>

MultiTracker3D::MultiTracker3D(double dt) : dt_(dt) {}

/**
Creates new traj trackers if the current measurement does not match to any existing trackers
*/
void MultiTracker3D::spawn(const Eigen::Vector3d& measurement){
    Filter f;
    f.obs.id = next_id_++;
    std::array<double,3> q{0.1, 0.1, 0.2};
    f.kf = std::make_unique<KalmanFilter3D>(measurement, dt_, 1.5, q);
    f.obs.history.push_back(measurement);
    filter_.push_back(std::move(f));
}

/**
Matches measurements to trackers, optionally spawning new trackers if the Mahalanobis distance is too far away.
Mahalanobis distance is a way of measuring how far a measurement is from a prediction whiel taking uncertainty into account.
d^2 = (z - \hat(z)).T * S^-1 (z - /hat(z)), essentially the residual x innovation covariance x residual.
*/
Step MultiTracker3D::step(const std::vector<Eigen::Vector3d>& measurements){

    // Predict Step
    for(auto& f: filter_) {
        f.kf->predict();
        f.age++;
    }

    std::vector<bool> matched(filter_.size(), false);
    std::vector<Eigen::Vector3d> unmatched_measurements;

    // Matching and Mahalanobis distance
    for(size_t z_idx = 0; z_idx < measurements.size(); ++z_idx){
        const auto& z = measurements[z_idx];
        double best_dist = gating_threshold_;
        int best_idx = -1;

        for(int i = 0; i < static_cast<int>(filter_.size()); i++){
            auto predicted = filter_[i].kf->position();
            Eigen::Vector3d r = (z - predicted);
            auto S = filter_[i].kf->innovationCovariance();
            double dist = r.transpose() * S.ldlt().solve(r);

            if(dist <= best_dist && !matched[i]){
                best_idx = i;
                best_dist = dist;
            }
        }

        if(best_idx >= 0){
            filter_[best_idx].kf->update(z);
            filter_[best_idx].obs.history.push_back(z);
            filter_[best_idx].missed = 0;
            matched[best_idx] = true;
        }
        else{
            unmatched_measurements.push_back(z);
        }
    }

    // Update state traj
    for(int i = 0; i < static_cast<int>(filter_.size()); i++){
        if(!matched[i]) {
            filter_[i].missed++;
        }
    }

    // Create new trackers 
    for(const auto& z : unmatched_measurements)
        spawn(z);

    // Prune dead trackers
    std::vector<int> dead_ids;
    for (const auto& f : filter_)
        if (f.missed > max_missed_)
            dead_ids.push_back(f.obs.id);

    filter_.erase(
        std::remove_if(filter_.begin(), filter_.end(),
            [&](const Filter& f) { return f.missed > max_missed_; }),
        filter_.end()
    );

    // Compile Obstacles
   std::vector<Obstacle> obstacles;
    obstacles.reserve(filter_.size());

    for (auto& f : filter_) {
        f.obs.predicted_trajectory.clear();
        auto traj = f.kf->predictTrajectory(num_predicted_steps_);
        for (auto& s : traj)
            f.obs.predicted_trajectory.push_back(s);

        if (f.age >= min_age_) {
            obstacles.push_back(f.obs);
        }
    }

    return Step{ obstacles, dead_ids };
}