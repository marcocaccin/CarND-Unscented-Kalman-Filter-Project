#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {

  // Create non-unitialised object
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  ///* Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // augmented sigma point matrix
  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // Weights of sigma points
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_.fill(0.5 / (n_aug_ + lambda_));
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2.0; //!!!
  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5; // !!!

  // Measurement noises: set by measurement device manufacturer
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;
  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;
  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;
  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;
  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // NIS radar
  NIS_radar_ = 0;
  // NIS laser
  NIS_laser_ = 0;
}

UKF::~UKF() {}


/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    x_.fill(0.0);
    P_.fill(0.0);
    P_(2,2) = 1.0;
    P_(3,3) = 1.0;
    P_(4,4) = 1.0;

    if (use_radar_ &&
        meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // Unpack measurement
      float rho = meas_package.raw_measurements_(0);
      float phi = meas_package.raw_measurements_(1);
      // Pour measured values into corresponding components of the state vector:
      // position and angle are available
      x_(0) = rho * cos(phi);
      x_(1) = rho * sin(phi);
      x_(3) = phi;
      // Initialise the state covariance matrix according to the
      // measurement covariances
      P_(0,0) = std_radr_*std_radr_;
      P_(1,1) = std_radr_*std_radr_;
      P_(3,3) = std_radphi_*std_radphi_;
    }
    else if (use_laser_ &&
             meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // Fill in only position
      x_(0) = meas_package.raw_measurements_(0);
      x_(1) = meas_package.raw_measurements_(1);

      P_(0,0) = std_laspx_*std_laspx_;
      P_(1,1) = std_laspy_*std_laspy_;
    }
    else {
      return;
    }

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
  *  Prediction
  ****************************************************************************/

  //compute the time elapsed between the current and previous measurements
  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  if (use_radar_ &&
      meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    Prediction(dt);
    UpdateRadar(meas_package);
  }
  else if (use_laser_ &&
           meas_package.sensor_type_ == MeasurementPackage::LASER) {
    Prediction(dt);
    UpdateLidar(meas_package);
  }
  return;
}


/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  // Evaluate sigma points
  AugmentedSigmaPoints();
  // Predict sigma points
  SigmaPointPrediction(delta_t);
  // Predict state and state covariance matrix
  PredictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  MatrixXd R = MatrixXd(2, 2);
  R << std_laspx_*std_laspx_, 0,
       0, std_laspy_*std_laspy_;

  MatrixXd H = MatrixXd(2, n_x_);
  H << 1, 0, 0, 0, 0,
       0, 1, 0, 0, 0;

  VectorXd z = meas_package.raw_measurements_;
  VectorXd y = z - H * x_;

  MatrixXd PHt = P_ * H.transpose();
  MatrixXd S = H * PHt + R;
  MatrixXd Si = S.inverse();
  MatrixXd K = PHt * Si;

  x_ += K * y;
  P_ = (MatrixXd::Identity(n_x_, n_x_) - K * H) * P_;

  NIS_laser_ = y.transpose() * Si * y;
}


/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  // Create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(3, 2 * n_aug_ + 1);

  // Transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {

    // Extract values for better readibility
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v   = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // Measurement model
    Zsig(0, i) = sqrt(p_x*p_x + p_y*p_y); // r
    Zsig(1, i) = atan2(p_y, p_x); // phi
    Zsig(2, i) = (p_x * v1 + p_y * v2) / Zsig(0, i); // r_dot
  }

  // Mean predicted measurement
  VectorXd z_pred = VectorXd(3);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
      z_pred += weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(3, 3);
  S.fill(0.0);
  // Cross correlation matrix Tc
  MatrixXd Tc = MatrixXd(n_x_, 3);
  Tc.fill(0.0);

  // Calculate measurement covariance matrix and cross correlation matrix
  // in the same loop
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    NormaliseAngle(z_diff(1));

    S += weights_(i) * z_diff * z_diff.transpose();

    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    NormaliseAngle(x_diff(3));

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }
  // Add measurement noise to measurement covariance matrix (diagonal only)
  S(0, 0) += std_radr_*std_radr_;
  S(1, 1) += std_radphi_*std_radphi_;
  S(2, 2) += std_radrd_*std_radrd_;


  // Get the current measurement (rho, phi, rho_dot)
  VectorXd z = meas_package.raw_measurements_;
  // Residual
  VectorXd y = z - z_pred;
  NormaliseAngle(y(1));

  MatrixXd Si = S.inverse();
  // Kalman gain K
  MatrixXd K = Tc * Si;

  // Update state mean and covariance matrix
  x_ += K * y;
  P_ -= K * S * K.transpose();

  NIS_radar_ = y.transpose() * Si * y;
}


void UKF::AugmentedSigmaPoints(void) {
  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  // Fill in augmented mean state
  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;

  // Fill in augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_*std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_*std_yawdd_;

  //create square root matrix
  MatrixXd A = P_aug.llt().matrixL();
  A *= sqrt(lambda_ + n_aug_);
  //create augmented sigma points
  Xsig_aug_.col(0)  = x_aug;
  for (int i = 0; i < n_aug_; i++)
  {
    Xsig_aug_.col(i + 1)       = x_aug + A.col(i);
    Xsig_aug_.col(i + 1 + n_aug_) = x_aug - A.col(i);
  }
}


void UKF::SigmaPointPrediction(double delta_t) {

  // Repeated calculations
  double delta_t2 = delta_t * delta_t;

  //predict sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    //extract values for better readability
    double p_x      = Xsig_aug_(0, i);
    double p_y      = Xsig_aug_(1, i);
    double v        = Xsig_aug_(2, i);
    double yaw      = Xsig_aug_(3, i);
    double yawd     = Xsig_aug_(4, i);
    double nu_a     = Xsig_aug_(5, i);
    double nu_yawdd = Xsig_aug_(6, i);

    // Repeated calculations
    double cos_yaw = cos(yaw);
    double sin_yaw = sin(yaw);

    //predicted state values
    double px_p, py_p;
    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin_yaw);
        py_p = p_y + v / yawd * (cos_yaw - cos(yaw + yawd * delta_t));
    }
    else {
        px_p = p_x + v * delta_t * cos_yaw;
        py_p = p_y + v * delta_t * sin_yaw;
    }
    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5 * nu_a * delta_t2 * cos_yaw;
    py_p = py_p + 0.5 * nu_a * delta_t2 * sin_yaw;
    v_p = v_p + nu_a * delta_t;
    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t2;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }
}


void UKF::PredictMeanAndCovariance(void) {

  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_ += weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    NormaliseAngle(x_diff(3));

    P_ += weights_(i) * x_diff * x_diff.transpose() ;
  }
}


/*
* A helper method to normalise an angle in the range (-pi, pi].
*/
void NormaliseAngle(double& angle) {
    while (angle >    M_PI) angle -= 2. * M_PI;
    while (angle <= - M_PI) angle += 2. * M_PI;
}


/*
Expected results on the simulator:

 X: 0.0712
 Y: 0.0811
VX: 0.3304
VY: 0.2452
*/
