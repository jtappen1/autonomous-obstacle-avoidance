#include "final_project/tracker/multi_tracker.hpp"
#include <memory>
#include <algorithm>

MultiTracker3D::MultiTracker3D(double dt) : dt_(dt) {}

// ─────────────────────────────────────────────────────────────────────────────
void MultiTracker3D::spawn(const Eigen::Vector3d& measurement){
    Trajectory t;
    t.r.id = next_id_++;
    std::array<double,3> q{0.1, 0.1, 0.2};
    t.kf = std::make_unique<KalmanFilter3D>(measurement, dt_, 1.0, q);
    t.r.history.push_back(measurement);
    traj_.push_back(std::move(t));
}

// ─────────────────────────────────────────────────────────────────────────────
StepResult MultiTracker3D::step(const std::vector<Eigen::Vector3d>& measurements){
    // ── Predict ──────────────────────────────────────────────────────────
    for(auto& t: traj_) {
        t.kf->predict();
        t.age++;
    }

   
    std::vector<bool> matched(traj_.size(), false);
    std::vector<Eigen::Vector3d> unmatched_measurements;

    for(size_t z_idx = 0; z_idx < measurements.size(); ++z_idx){
        const auto& z = measurements[z_idx];
        double best_dist = gating_threshold_;
        int best_idx = -1;

        for(int i = 0; i < static_cast<int>(traj_.size()); i++){
            auto predicted = traj_[i].kf->position();
            Eigen::Vector3d r = (z - predicted);
            auto S = traj_[i].kf->innovationCovariance();
            double dist = r.transpose() * S.ldlt().solve(r);

            if(dist <= best_dist && !matched[i]){
                best_idx = i;
                best_dist = dist;
            }
        }

        if(best_idx >= 0){
            traj_[best_idx].kf->update(z);
            traj_[best_idx].r.history.push_back(z);
            traj_[best_idx].missed = 0;
            matched[best_idx] = true;
        }
        else{
            unmatched_measurements.push_back(z);
        }
    }

    for(int i = 0; i < static_cast<int>(traj_.size()); i++){
        if(!matched[i]) {
            traj_[i].missed++;
        }
    }

    // ── Spawn deferred new tracks ─────────────────────────────────────────
    for(const auto& z : unmatched_measurements)
        spawn(z);

    // ── Prune dead tracks ─────────────────────────────────────────────────
    std::vector<int> dead_ids;
    for (const auto& t : traj_)
        if (t.missed > max_missed_)
            dead_ids.push_back(t.r.id);

    traj_.erase(
        std::remove_if(traj_.begin(), traj_.end(),
            [&](const Trajectory& t) { return t.missed > max_missed_; }),
        traj_.end()
    );

    // ── Build return value ────────────────────────────────────────────────
    std::vector<TrajectoryHistory> traj_histories;
    traj_histories.reserve(traj_.size());
    std::transform(traj_.begin(), traj_.end(),
        std::back_inserter(traj_histories),
        [](const auto& t) { return t.r; });

    return StepResult{ traj_histories, dead_ids };
}