#ifndef RIDGE_REGRESSION_H
#define RIDGE_REGRESSION_H
#include "Eigen\Dense"
#include <string>
using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::VectorXd;

class RR
{
public:
	void train(const MatrixXd& data, double lamda=0.1);
	MatrixXd predict(const MatrixXd& X);

private: 
	//trained parameters 
	VectorXd beta;
	double offset;
};
#endif