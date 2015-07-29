#ifndef LEAST_ABSOLUTE_REGRESSION_H
#define LEAST_ABSOLUTE_REGRESSION_H
#include "Eigen\Dense"
#include <string>
using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::VectorXd;

class LAR
{
public:
	void train(const MatrixXd& data);
	MatrixXd predict(const MatrixXd& X);

private: 
	//trained parameters 
	VectorXd beta;
	double offset;
};
#endif