#include "LeastAbsoluteRegression.h"
#include "Eigen\Dense"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

using namespace std;
using namespace Eigen;
#define LINE cout<<string(65,'-')<<endl;

void LAR::train(const MatrixXd& data )
{ 
	LINE; LINE;
	cout << "Data Dimension : " << data.rows() << " X " << data.cols() << endl;
	LINE;
	cout << endl;

	//Training data
	double trainRatio = 2.8 / 3;
	int breakPoint = (int)(data.rows() * trainRatio), col = data.cols(), row = data.rows();

	MatrixXd X_(breakPoint, col);
	X_ << data.block(0, 0, breakPoint, col - 1), MatrixXd::Ones(breakPoint, 1);
	MatrixXd Y = data.block(0, col - 1, breakPoint, 1);

	//testing
	MatrixXd Xts(row - breakPoint, col);
	Xts = data.block(breakPoint, 0, row - breakPoint, col - 1), MatrixXd::Ones(row - breakPoint, 1);
	MatrixXd Yts = data.block(breakPoint, col - 1, row - breakPoint, 1);

	//Least absolute error regression 

	//initialize the parameter 
	VectorXd beta_new = VectorXd::Zero(col);
	VectorXd beta_old = VectorXd::Ones(col);

	cout << "Iteratively weighted least square regression:" << endl << endl;
	int i = 0;
	while ((beta_new - beta_old).norm()>0.01)
	{
		beta_old = beta_new;
		//Parameter estiamtion -- iteratively weighted least square
		VectorXd weight = 1 / (Y - X_*beta_old).array().abs();
		MatrixXd C = weight.asDiagonal();
		beta_new = (X_.transpose()*C*X_).inverse()*X_.transpose()*C*Y;
		cout << "Training iteration: " << setw(3) << ++i << " , current coefficients: " << beta_new.transpose() << endl;
	} 
	//prediction on test set 

	LINE;
	cout << "Result: " << endl << endl;
	beta = beta_new.head(beta_new.size()-1);
	cout << " Coefficients for X are: " << beta << endl;
	offset = beta_new(beta_new.size() - 1);
	cout << " Offset term is: " << offset << endl;
	LINE; LINE;
}
 
MatrixXd LAR::predict(const MatrixXd& X)
{
	return (X*beta).array() + offset;
}