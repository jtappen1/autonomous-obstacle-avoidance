#include "final_project/tracker/UKF.hpp"

#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <Eigen/Eigenvalues> // Required for eigenvalue clamping

namespace final_project::tracker {

namespace {
constexpr double kYawRateEps = 1e-4;
}

UKF_CTRV::UKF_CTRV(const Vec2& z0,
                   const Mat2& R0,
                   double dt,
                   double std_a,
                   double std_yawdd)
    : std_a_(std_a), std_yawdd_(std_yawdd), last_dt_(dt) {
    x_.setZero();
    x_(0) = z0.x();
    x_(1) = z0.y();

    P_.setZero();
    P_.block<2, 2>(0, 0) = R0;
    P_(2, 2) = 25.0;
    P_(3, 3) = M_PI * M_PI;
    P_(4, 4) = 1.0;

    initializeWeights();
    Xsig_pred_.setZero();
}
double UKF_CTRV::normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

void UKF_CTRV::initializeWeights() {
    weights_.setConstant(0.5 / (lambda_ + static_cast<double>(kAugDim)));
    weights_(0) = lambda_ / (lambda_ + static_cast<double>(kAugDim));
}

void UKF_CTRV::predictSigmaPoints(double dt) {
    Vec7 x_aug = Vec7::Zero();
    x_aug.head<kStateDim>() = x_;

    Mat7 P_aug = Mat7::Zero();
    P_aug.topLeftCorner<kStateDim, kStateDim>() = P_;
    P_aug(5, 5) = std_a_ * std_a_;
    P_aug(6, 6) = std_yawdd_ * std_yawdd_;

    // Force symmetry before factorization
    P_aug = 0.5 * (P_aug + P_aug.transpose());

    Mat7 L;
    bool success = false;
    double jitter = 1e-9;

    for (int attempt = 0; attempt < 8; ++attempt) {
        Eigen::LLT<Mat7> llt(P_aug);
        if (llt.info() == Eigen::Success) {
            L = llt.matrixL();
            success = true;
            break;
        }
        P_aug.diagonal().array() += jitter;
        jitter *= 10.0;
    }

    if (!success) {
        throw std::runtime_error("UKF_CTRV: LLT failed while building augmented sigma points");
    }

    Eigen::Matrix<double, kAugDim, kSigmaCount> Xsig_aug;
    Xsig_aug.col(0) = x_aug;
    const double scale = std::sqrt(lambda_ + static_cast<double>(kAugDim));
    for (int i = 0; i < kAugDim; ++i) {
        Xsig_aug.col(i + 1) = x_aug + scale * L.col(i);
        Xsig_aug.col(i + 1 + kAugDim) = x_aug - scale * L.col(i);
    }

    for (int i = 0; i < kSigmaCount; ++i) {
        const double px = Xsig_aug(0, i);
        const double py = Xsig_aug(1, i);
        const double v = Xsig_aug(2, i);
        const double yaw = Xsig_aug(3, i);
        const double yaw_rate = Xsig_aug(4, i);
        const double nu_a = Xsig_aug(5, i);
        const double nu_yawdd = Xsig_aug(6, i);

        double px_p = px;
        double py_p = py;

        if (std::fabs(yaw_rate) > kYawRateEps) {
            px_p += (v / yaw_rate) * (std::sin(yaw + yaw_rate * dt) - std::sin(yaw));
            py_p += (v / yaw_rate) * (-std::cos(yaw + yaw_rate * dt) + std::cos(yaw));
        } else {
            px_p += v * std::cos(yaw) * dt;
            py_p += v * std::sin(yaw) * dt;
        }

        double v_p = v;
        double yaw_p = yaw + yaw_rate * dt;
        double yaw_rate_p = yaw_rate;

        px_p += 0.5 * dt * dt * std::cos(yaw) * nu_a;
        py_p += 0.5 * dt * dt * std::sin(yaw) * nu_a;
        v_p += dt * nu_a;
        yaw_p += 0.5 * dt * dt * nu_yawdd;
        yaw_rate_p += dt * nu_yawdd;

        Xsig_pred_(0, i) = px_p;
        Xsig_pred_(1, i) = py_p;
        Xsig_pred_(2, i) = v_p;
        Xsig_pred_(3, i) = normalizeAngle(yaw_p);
        Xsig_pred_(4, i) = yaw_rate_p;
    }
}

void UKF_CTRV::recoverMeanAndCovariance() {
    x_.setZero();

    // Linear parts
    for (int i = 0; i < kSigmaCount; ++i) {
        x_(0) += weights_(i) * Xsig_pred_(0, i);
        x_(1) += weights_(i) * Xsig_pred_(1, i);
        x_(2) += weights_(i) * Xsig_pred_(2, i);
        x_(4) += weights_(i) * Xsig_pred_(4, i);
    }

    // Circular mean for yaw
    double sin_sum = 0.0;
    double cos_sum = 0.0;
    for (int i = 0; i < kSigmaCount; ++i) {
        sin_sum += weights_(i) * std::sin(Xsig_pred_(3, i));
        cos_sum += weights_(i) * std::cos(Xsig_pred_(3, i));
    }
    x_(3) = std::atan2(sin_sum, cos_sum);

    P_.setZero();
    for (int i = 0; i < kSigmaCount; ++i) {
        Vec5 dx = Xsig_pred_.col(i) - x_;
        dx(3) = normalizeAngle(dx(3));
        P_ += weights_(i) * dx * dx.transpose();
    }

    // Force symmetry and strictly enforce positive-definiteness
    P_ = 0.5 * (P_ + P_.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, kStateDim, kStateDim>> es(P_);
    Eigen::VectorXd evals = es.eigenvalues();
    for (int i = 0; i < evals.size(); ++i) {
        evals(i) = std::max(evals(i), 1e-6); 
    }
    P_ = es.eigenvectors() * evals.asDiagonal() * es.eigenvectors().transpose();
}

void UKF_CTRV::predict(double dt) {
    last_dt_ = std::max(dt, 1e-3);
    predictSigmaPoints(last_dt_);
    recoverMeanAndCovariance();
}

UKF_CTRV::Vec2 UKF_CTRV::velocity() const {
    return Vec2(x_(2) * std::cos(x_(3)), x_(2) * std::sin(x_(3)));
}

UKF_CTRV::Mat2 UKF_CTRV::innovationCovariance(const Mat2& R) const {
    Mat2 S = P_.topLeftCorner<2, 2>() + R;
    S += 1e-6 * Mat2::Identity();
    return S;
}

void UKF_CTRV::updatePosition(const Vec2& z, const Mat2& R) {
    ZSigmaMat Zsig;
    for (int i = 0; i < kSigmaCount; ++i) {
        Zsig.col(i) = Xsig_pred_.block<2, 1>(0, i);
    }

    Vec2 z_pred = Vec2::Zero();
    for (int i = 0; i < kSigmaCount; ++i) {
        z_pred += weights_(i) * Zsig.col(i);
    }

    Mat2 S = Mat2::Zero();
    for (int i = 0; i < kSigmaCount; ++i) {
        Vec2 z_diff = Zsig.col(i) - z_pred;
        S += weights_(i) * z_diff * z_diff.transpose();
    }
    S += R;
    S += 1e-6 * Mat2::Identity();
    S = 0.5 * (S + S.transpose());

    Eigen::Matrix<double, kStateDim, 2> Tc =
        Eigen::Matrix<double, kStateDim, 2>::Zero();
    for (int i = 0; i < kSigmaCount; ++i) {
        Vec2 z_diff = Zsig.col(i) - z_pred;
        Vec5 x_diff = Xsig_pred_.col(i) - x_;
        x_diff(3) = normalizeAngle(x_diff(3));
        Tc += weights_(i) * x_diff * z_diff.transpose();
    }

    Eigen::LDLT<Mat2> ldlt(S);
    if (ldlt.info() != Eigen::Success) {
        throw std::runtime_error("UKF_CTRV: LDLT failed in updatePosition");
    }

    const Eigen::Matrix<double, kStateDim, 2> K =
        Tc * ldlt.solve(Mat2::Identity());

    Vec2 z_diff = z - z_pred;
    x_ += K * z_diff;
    x_(3) = normalizeAngle(x_(3));

    P_ -= K * S * K.transpose();
    
    // Force symmetry and strictly enforce positive-definiteness
    P_ = 0.5 * (P_ + P_.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, kStateDim, kStateDim>> es(P_);
    Eigen::VectorXd evals = es.eigenvalues();
    for (int i = 0; i < evals.size(); ++i) {
        evals(i) = std::max(evals(i), 1e-6); 
    }
    P_ = es.eigenvectors() * evals.asDiagonal() * es.eigenvectors().transpose();

    // Do NOT call predictSigmaPoints(last_dt_) here.
    // The next predict() call should generate the next sigma points.
}

std::vector<UKFStatePrediction> UKF_CTRV::predictTrajectory(int steps, double dt) const {
    std::vector<UKFStatePrediction> out;
    out.reserve(std::max(steps, 0));

    UKF_CTRV clone = *this;
    for (int i = 0; i < steps; ++i) {
        clone.predict(dt);
        UKFStatePrediction pred;
        pred.position = clone.position();
        pred.velocity = clone.velocity();
        pred.speed = clone.speed();
        pred.yaw = clone.yaw();
        pred.yaw_rate = clone.yawRate();
        pred.covariance = clone.covariance();
        out.push_back(pred);
    }
    return out;
}

}  // namespace final_project::tracker