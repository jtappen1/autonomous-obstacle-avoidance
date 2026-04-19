#include "final_project/tracker/kalman_filter_3d.hpp"
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
KalmanFilter3D::KalmanFilter3D(
    const Vec3& measurement,
    double dt, 
    double sigma_a,
    const std::array<double, 3>& sigma_z)
{
    // State Vector
    x_.head<3>() = measurement;
    x_.tail<3>().setZero();

    // Covariance
    P_.setIdentity();
    P_.block<3,3>(0,0) *= 1.0;   // position fairly confident
    P_.block<3,3>(3,3) *= 100.0; // velocity very uncertain

    // State transition  F = I + dt * [0  I; 0  0]
    F_ = Mat6::Identity();
    F_(0, 3) = dt;
    F_(1, 4) = dt;
    F_(2, 5) = dt;

    // Process noise
    Q_ = buildQ(dt, sigma_a);

    // Measurement matrix — we observe [x, y, z] directly
    H_.setZero();
    H_(0, 0) = 1.0;
    H_(1, 1) = 1.0;
    H_(2, 2) = 1.0;

    // Measurement noise
    R_ = Mat3::Zero();
    for (int i = 0; i < 3; ++i)
        R_(i, i) = sigma_z[i] * sigma_z[i];
}

// ─────────────────────────────────────────────────────────────────────────────
KalmanFilter3D::Mat6 KalmanFilter3D::buildQ(double dt, double sigma_a)
{
    const double q   = sigma_a * sigma_a;
    const double dt2 = dt  * dt;
    const double dt3 = dt2 * dt;
    const double dt4 = dt3 * dt;

    // 1-D process noise block
    //  [dt4/4  dt3/2]
    //  [dt3/2  dt2  ]
    const double q00 = dt4 / 4.0 * q;
    const double q01 = dt3 / 2.0 * q;
    const double q11 = dt2       * q;

    Mat6 Q = Mat6::Zero();
    for (int i = 0; i < 3; ++i) {
        Q(i,   i  ) = q00;
        Q(i,   i+3) = q01;
        Q(i+3, i  ) = q01;   // symmetric
        Q(i+3, i+3) = q11;
    }
    return Q;
}
// ─────────────────────────────────────────────────────────────────────────────
KalmanFilter3D::Mat3 KalmanFilter3D::innovationCovariance() const {
    Mat3 S = H_ * P_ * H_.transpose() + R_;
    S += 1e-6 * Mat3::Identity();
    return S;
}

// ─────────────────────────────────────────────────────────────────────────────
void KalmanFilter3D::predict()
{
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
}

// ─────────────────────────────────────────────────────────────────────────────
void KalmanFilter3D::update(const Vec3& z)
{
    // Innovation
    const Vec3 y = z - H_ * x_;

    // Innovation covariance
    const Mat3 S = innovationCovariance();

    // Kalman gain  K = P H^T S^{-1}
    const Mat6x3 PHt = P_ * H_.transpose();
    const Mat6x3 K = PHt * S.ldlt().solve(Mat3::Identity());
    // State update
    x_ += K * y;

    // Joseph-form covariance update (numerically stable)
    const Mat6 I_KH = Mat6::Identity() - K * H_;
    P_ = I_KH * P_ * I_KH.transpose() + K * R_ * K.transpose();

    
}
