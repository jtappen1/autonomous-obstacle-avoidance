#pragma once

#include <Eigen/Dense>
#include <vector>

namespace final_project::tracker {

struct UKFStatePrediction {
    Eigen::Vector2d position = Eigen::Vector2d::Zero();
    Eigen::Vector2d velocity = Eigen::Vector2d::Zero();
    double speed = 0.0;
    double yaw = 0.0;
    double yaw_rate = 0.0;
    Eigen::Matrix<double, 5, 5> covariance = Eigen::Matrix<double, 5, 5>::Zero();
};

class UKF_CTRV {
public:
    using Vec2 = Eigen::Vector2d;
    using Mat2 = Eigen::Matrix2d;
    using Vec5 = Eigen::Matrix<double, 5, 1>;   // [px, py, v, yaw, yaw_rate]
    using Mat5 = Eigen::Matrix<double, 5, 5>;

    UKF_CTRV(const Vec2& z0,
             const Mat2& R0,
             double dt,
             double std_a = 2.0,
             double std_yawdd = 1.0);

    void predict(double dt);
    void updatePosition(const Vec2& z, const Mat2& R);

    Vec2 position() const { return x_.head<2>(); }
    Vec2 velocity() const;
    double speed() const { return x_(2); }
    double yaw() const { return x_(3); }
    double yawRate() const { return x_(4); }
    const Vec5& state() const { return x_; }
    const Mat5& covariance() const { return P_; }

    Mat2 innovationCovariance(const Mat2& R) const;
    std::vector<UKFStatePrediction> predictTrajectory(int steps, double dt) const;

private:
    static constexpr int kStateDim = 5;
    static constexpr int kAugDim = 7;
    static constexpr int kSigmaCount = 2 * kAugDim + 1;

    using Vec7 = Eigen::Matrix<double, kAugDim, 1>;
    using Mat7 = Eigen::Matrix<double, kAugDim, kAugDim>;
    using SigmaMat = Eigen::Matrix<double, kStateDim, kSigmaCount>;
    using ZSigmaMat = Eigen::Matrix<double, 2, kSigmaCount>;

    static double normalizeAngle(double angle);
    void initializeWeights();
    void predictSigmaPoints(double dt);
    void recoverMeanAndCovariance();

    Vec5 x_ = Vec5::Zero();
    Mat5 P_ = Mat5::Identity();
    SigmaMat Xsig_pred_ = SigmaMat::Zero();
    Eigen::Matrix<double, kSigmaCount, 1> weights_ =
        Eigen::Matrix<double, kSigmaCount, 1>::Zero();

    double std_a_ = 2.0;
    double std_yawdd_ = 1.0;
    double lambda_ = 3.0 - static_cast<double>(kAugDim);
    double last_dt_ = 1.0 / 15.0;
};

}  // namespace final_project::tracker