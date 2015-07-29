#ifndef NAIVE_BAYES_CLASSIFIER_H
#define NAIVE_BAYES_CLASSIFIER_H
#include "Eigen\Dense"
#include <string>
using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::VectorXd;

class NB
{
public:
	NB(bool zeroOne_=true) :zeroOne(zeroOne_){} //specify 0/1 input or -1/1 input 
	void train(const MatrixXd& data, double trainRatio = 2.0 / 3);
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
	double pi_0;
	double pi_1;
	MatrixXd g_0; //2*n
	MatrixXd g_1; //2*n
};
#endif