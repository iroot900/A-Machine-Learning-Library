#include "RidgeRegression.h"
#include "Eigen\Dense"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

using namespace std;
using namespace Eigen;
#define LINE cout<<string(65,'-')<<endl;

void RR::train(const MatrixXd& data, double lamda)
{ 
	LINE; LINE;
	cout << "Data Dimension : " << data.rows() << " X " << data.cols() << endl;
	LINE;
	cout << endl;

	//Training data
	int col = data.cols(), row = data.rows();
	MatrixXd X = data.block(0, 0, row, col - 1);
	MatrixXd Y = data.col(col - 1);

	//most of the time, the data dimensin wouldn't be so high
	//(X'X+n*lamda) which is (d*d) is convex and inversible computatioanly 

	//Ridge regression
	//center data; 
	VectorXd x_u = X.colwise().mean();
	for (int i = 0; i < row; ++i) X.row(i) = X.row(i) - x_u;
	double Y_u = Y.mean();
	Y = Y.array() - Y_u;

	//Parameter estiamtion 
	beta = (X.transpose()*X + row*lamda* MatrixXd::Identity(col - 1, col - 1)).inverse()*X.transpose()*Y;
	offset = Y_u - (beta*x_u.transpose())(0, 0);

	//value of objective function:
	double obj = ((1 / row)* (Y - X*beta).transpose()*(Y - X*beta) + lamda*beta.transpose()*beta)(0, 0);

	LINE;
	cout << "Result: " << endl;
	cout << " Coefficients for X are: " << beta.transpose() << endl;
	cout << " Offset term is: " << offset << endl;
	cout << " Value for objective function is: " << obj << endl;
	LINE; LINE;
}
 
MatrixXd RR::predict(const MatrixXd& X)
{
	return (X*beta).array() + offset;
}