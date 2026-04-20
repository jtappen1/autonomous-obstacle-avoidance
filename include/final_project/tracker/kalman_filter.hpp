#pragma once
#include <array>
#include <Eigen/Dense>


struct PredictedState {
    Eigen::Vector3d position;
    Eigen::Vector3d velocity;
    Eigen::Matrix<double, 6, 6> covariance;
};

class KalmanFilter3D {
public:
    using Vec6 = Eigen::Matrix<double, 6, 1>;
    using Mat6 = Eigen::Matrix<double, 6, 6>;
    using Vec3 = Eigen::Matrix<double, 3, 1>;
    using Mat3 = Eigen::Matrix<double, 3, 3>;
    using Mat3x6 = Eigen::Matrix<double, 3, 6>;
    using Mat6x3 = Eigen::Matrix<double, 6, 3>;
 
    /**
     * @param dt      Time step in seconds.
     * @param sigma_a Process noise (acceleration std dev).
     * @param sigma_z Measurement noise std devs [sx, sy, sz].
     */
    KalmanFilter3D(const Vec3& measurement, double dt, double sigma_a,
                   const std::array<double, 3>& sigma_z);
 
    /// Predict step — propagates state and covariance forward by dt.
    void predict();
 
    /// Update step — fuses a 3-D position measurement.
    void update(const Vec3& z);
    
    std::vector<PredictedState> predictTrajectory(int steps) const;
 
    Vec3 position() const { return x_.head<3>(); }
    Vec3 velocity() const { return x_.tail<3>(); }
    const Vec6& state()  const { return x_; }
    const Mat6& F()      const { return F_; }
    Mat3 innovationCovariance() const;
 
private:
    static Mat6 buildQ(double dt, double sigma_a);
 
    Vec6    x_;   ///< State  [x, y, z, vx, vy, vz]
    Mat6    P_;   ///< Covariance
    Mat6    F_;   ///< State transition
    Mat6    Q_;   ///< Process noise covariance
    Mat3x6  H_;   ///< Measurement matrix
    Mat3    R_;   ///< Measurement noise covariance
};