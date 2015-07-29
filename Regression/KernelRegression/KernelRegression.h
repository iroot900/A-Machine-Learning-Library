#ifndef KERNEL_REGRESSION_H
#define KERNEL_REGRESSION_H
#include "Eigen\Dense"
#include <string>
using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::VectorXd;

//only support gaussian kernel currently 
class KR
{
public:
	void train(const MatrixXd& data, double lamda = 0.001, double sigma = 1.5);
	MatrixXd predict(const MatrixXd& X);

private: 
	//trained parameters 
	
	double lamda_;
	double sigma_;
	//needed for prediction purpose
	MatrixXd K_tr; //kernel Matrix
	MatrixXd X_tr;
	MatrixXd Y_tr;
};
#endif