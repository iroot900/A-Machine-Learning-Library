#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H
#include "Eigen\Dense"
#include <string>
using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::VectorXd;

class LR
{
public:
	LR(bool zeroOne_):zeroOne(zeroOne_){} //specify 0/1 input or -1/1 input 
	void train(const MatrixXd& data, double trainRatio = 2.0 / 3, double lamda = 10, double alpha = 0.0001);
	MatrixXd predict(const MatrixXd& X); //predic for multiple input

private:

	bool zeroOne=true;
	//Error rate on training
	double ErrorTr;
	//Error rate on testing 
	double ErrorTs;
	//label for training set
	MatrixXd labelTr;
	//label for testing set
	MatrixXd labelTs;

	//trained parameters
	MatrixXd theta;
};
#endif