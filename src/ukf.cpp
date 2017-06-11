#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

#define EPS 0.001

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;

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

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  n_x_ = 5;
  
  n_aug_ = 7;
  
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  
  weights_ = VectorXd(2 * n_aug_ + 1);
  
  lambda_ = 3 - n_aug_;
  
  is_initialized_ = false;
  
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  
  
  if(!is_initialized_){
    time_us_ = meas_package.timestamp_;
    
    P_ << 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0,
          0, 0, 1, 0, 0,
          0, 0, 0, 1, 0,
          0, 0, 0, 0, 1;
    
    if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
      double rho = meas_package.raw_measurements_[0];
      double theta = meas_package.raw_measurements_[1];
      
      double p_x = rho * cos(theta);
      double p_y = rho * sin(theta);
      double v = 0.0;
      double psi = 0.0;
      double psi_dot = 0.0;
      
      x_ << p_x, p_y, v, psi, psi_dot;
    }else{
      double p_x = meas_package.raw_measurements_[0];
      double p_y = meas_package.raw_measurements_[1];
      double v = 0.0;
      double psi = 0.0;
      double psi_dot = 0.0;
      
      x_ << p_x, p_y,  v, psi, psi_dot;
    }
    is_initialized_ = true;
    return;
  }
  
  float delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
  time_us_ = meas_package.timestamp_;
  
  Prediction(delta_t);
  
  if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
    UpdateRadar(meas_package);
    
  }else{
    UpdateLidar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  //create augmented mean vector
  VectorXd x_aug = VectorXd(7);
  x_aug.fill(0.0);
  
  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);
  P_aug.fill(0.0);
  
  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug.fill(0.0);

  
  //create augmented mean state
  x_aug.head(n_x_) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;
  
  //create augmented covariance matrix
  MatrixXd Q = MatrixXd(2,2);
  Q << std_a_ * std_a_, 0,
  0, std_yawdd_ * std_yawdd_;
  
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(2, 2) = Q;
  
  //create square root matrix
  
  MatrixXd P_aug_sqrt = P_aug.llt().matrixL();
  //create augmented sigma points
  
  Xsig_aug.col(0) = x_aug;
  
  for(int i = 0; i < n_aug_; i++){
    Xsig_aug.col(i+1)     = x_aug + sqrt(lambda_+n_aug_) * P_aug_sqrt.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * P_aug_sqrt.col(i);
  }
  
  //predict state
  VectorXd curr_sigm_point = VectorXd(7);
  VectorXd curr_state = VectorXd(5);
  VectorXd new_state = VectorXd(5);
  
  for(int i = 0; i < Xsig_pred_.cols(); i++){
    curr_sigm_point = Xsig_aug.col(i);
    curr_state = curr_sigm_point.head(5);
    double v = curr_sigm_point(2);
    double psi = curr_sigm_point(3);
    double psi_dot = curr_sigm_point(4);
    double miu_a = curr_sigm_point(5);
    double miu_psi_dot_dot = curr_sigm_point(6);
    double delta_t_pow = delta_t * delta_t;
    VectorXd noise = VectorXd(5);
    noise(0) = 0.5 * delta_t_pow * cos(psi) * miu_a;
    noise(1) = 0.5 * delta_t_pow * sin(psi) * miu_a;
    noise(2) = delta_t * miu_a;
    noise(3) = 0.5 * delta_t_pow * miu_psi_dot_dot;
    noise(4) = delta_t * miu_psi_dot_dot;
    if(psi_dot == 0){
      new_state(0) = v * cos(psi) * delta_t;
      new_state(1) = v * sin(psi) * delta_t;
      new_state(2) = 0;
      new_state(3) = psi_dot * delta_t;
      new_state(4) = 0;
    }else{
      double v_psi = v / psi_dot;
      double new_psi = psi + psi_dot * delta_t;
      new_state(0) = v_psi * (sin(new_psi) - sin(psi));
      new_state(1) = v_psi * (-cos(new_psi) + cos(psi));
      new_state(2) = 0;
      new_state(3) = psi_dot * delta_t;
      new_state(4) = 0;
    }
    
    new_state += noise;
    new_state += curr_state;
    Xsig_pred_.col(i) = new_state;
  }
  
  
  //set weights
  for(int i =0; i < weights_.rows(); i++) {
    if(i == 0){
      weights_(i) = lambda_ / (lambda_ + n_aug_);
    } else {
      weights_(i) = 1 / (2 * (lambda_ + n_aug_));
    }
  }
  
  //predicted state mean
  x_.fill(0.0);
  for(int j = 0; j < Xsig_pred_.cols(); j++){
    x_ += weights_(j) * Xsig_pred_.col(j);
  }
  
  //predict state covariance matrix
  P_.fill(0.0);
  for(int j = 0; j < Xsig_pred_.cols(); j++){
    VectorXd x_diff = Xsig_pred_.col(j) - x_;
    
    Normalize_angle(x_diff(3));

    P_ += weights_(j) * x_diff * x_diff.transpose() ;
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  int n_z = 2;
  
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  Zsig.fill(0.0);
  
  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    
    //measurement model
    Zsig(0,i) = Xsig_pred_(0,i);
    Zsig(1,i) = Xsig_pred_(1,i);
  }
  
  MatrixXd R = MatrixXd(n_z,n_z);
  R.fill(0.0);
  R(0,0) = std_laspx_*std_laspx_;
  R(1,1) = std_laspy_*std_laspy_;
  
  Update(Zsig, R, meas_package);
  
  }

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  int n_z = 3;
  
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  Zsig.fill(0.0);
  
  double p_x;
  double p_y;
  double v;
  double psi;
  double psi_dot;
  double sqrt_p_sum;
  //transform sigma points into measurement space
  for(int i = 0; i < Xsig_pred_.cols(); i++) {
    p_x = Xsig_pred_(0, i);
    p_y = Xsig_pred_(1, i);
    v = Xsig_pred_(2, i);
    psi = Xsig_pred_(3, i);
    psi_dot = Xsig_pred_(4, i);
    
    sqrt_p_sum = sqrt(p_x*p_x + p_y*p_y);
    
    Zsig(0, i) = sqrt_p_sum;
    Zsig(1, i) = atan2(p_y, p_x);
    Zsig(2, i) = ((p_x * cos(psi) * v) + (p_y * sin(psi) * v))/ sqrt_p_sum;
  }
  
  MatrixXd R = MatrixXd(n_z, n_z);
  R.fill(0.0);
  R(0,0) = std_radr_ * std_radr_;
  R(1,1) = std_radphi_ * std_radphi_;
  R(2,2) = std_radrd_ * std_radrd_;
  
  Update(Zsig, R, meas_package);
}

void UKF::Update(MatrixXd Zsig, MatrixXd R, MeasurementPackage meas_package){
  int n_z = Zsig.rows();
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  
  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);

  //calculate mean predicted measurement
  for(int i = 0; i < Zsig.cols(); i++){
    z_pred += weights_(i) * Zsig.col(i);
  }
  
  for(int i = 0; i < Zsig.cols(); i++){
    VectorXd z_diff = VectorXd(n_z);
    z_diff = Zsig.col(i) - z_pred;
    
    Normalize_angle(z_diff(1));
    
    S += weights_(i) * z_diff * z_diff.transpose();
  }
  S += R;
  
  //mesurements matrix
  VectorXd z = VectorXd(n_z);
  
  z << meas_package.raw_measurements_;
  
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  
  Tc.fill(0.0);
  //calculate cross correlation matrix
  for(int i = 0; i < Zsig.cols(); i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    Normalize_angle(x_diff(3));
    
    VectorXd z_diff = Zsig.col(i) - z_pred;
    Normalize_angle(z_diff(1));
    
    Tc += weights_(i) * x_diff * z_diff.transpose();
  }
  
  //calculate Kalman gain K;
  MatrixXd K = MatrixXd(n_x_, n_z);
  K = Tc * S.inverse();
  //update state mean and covariance matrix
  
  VectorXd z_diff = z - z_pred;
  Normalize_angle(z_diff(1));
  
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
  
  //calculate the Normalized Innovation Square
  if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
    NIS_RADAR_= z_diff.transpose() * S.inverse() * z_diff;
  }else{
    NIS_LASER_= z_diff.transpose() * S.inverse() * z_diff;
  }

}

void UKF::Normalize_angle(double &angle){
  while (angle> M_PI) angle-=2.*M_PI;
  while (angle<-M_PI) angle+=2.*M_PI;
}
