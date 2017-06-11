#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
  
  VectorXd rmse(4);
  rmse << 0,0,0,0;
  
  int size = estimations.size();
  
  if(size != ground_truth.size() || size == 0){
    return rmse;
  }
  
  for(int i=0; i < size; ++i){
    // ... your code here
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse += residual;
  }
  rmse = rmse/size;
  
  rmse = rmse.array().sqrt();
  std::cout << "RMSE" << std::endl;
  std::cout << rmse << std::endl;
  
  //return the result
  return rmse;

}
