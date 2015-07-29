#include "KernelRegression.h"
#include "Eigen\Dense"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

using namespace std;
using namespace Eigen;
#define LINE cout<<string(65,'-')<<endl;

void KR::train(const MatrixXd& data, double lamda, double sigma)
{ 
	lamda_ = lamda;
	sigma_ = sigma;

	LINE; LINE;
	cout << "Data Dimension : " << data.rows() << " X " << data.cols() << endl;
	LINE;
	cout << endl;

	//Training data
	double trainRatio = 2.5 / 3;
	int breakPoint = (int)(data.rows() * trainRatio), col = data.cols(), row = data.rows();

	MatrixXd X = data.block(0, 0, breakPoint, col - 1);
	MatrixXd Y = data.block(0, col - 1, breakPoint, 1);
	double y_u = Y.mean();
	MatrixXd Y_ = Y.array() - y_u;
	X_tr = X; Y_tr = Y;
	//testing
	MatrixXd Xts = data.block(breakPoint, 0, row - breakPoint, col - 1);
	MatrixXd Yts = data.block(breakPoint, col - 1, row - breakPoint, 1);

	//Kernel Ridge regression

	//inter product matrix 
	MatrixXd K_G(breakPoint, breakPoint);
	MatrixXd K_P(breakPoint, breakPoint);
	VectorXd Unitv = VectorXd::Ones(breakPoint);
	for (int i = 0; i < breakPoint; ++i)
	{
		//gaussian kernel
		K_G.row(i) = (-1 * (((Unitv*X.row(i)) - X).rowwise().squaredNorm() / (2 * sigma*sigma))).array().exp();
	}

	//centered inter product matrix 
	MatrixXd O = MatrixXd::Ones(breakPoint, breakPoint);
	O = O / breakPoint;
	MatrixXd K_G_ = K_G - K_G*O - O*K_G + O*K_G*O;
	K_tr = K_G_;
	cout << endl;
	LINE;

	//prediction on test set
	cout << "Prediction of Y on test set: " << endl << endl;
	cout << "y_pred(gaussian),   y_orig" << endl << endl;
	MatrixXd Y_2(Xts.rows(),2);
	Y_2 << predict(Xts), Yts;
	cout << Y_2 << endl;

	LINE; LINE;
	cout << endl;
	cout << "Result :" << endl;
	cout << "Gaussian Kernel Sigma = " << sigma << endl;
}

MatrixXd KR::predict(const MatrixXd& X)
{
	MatrixXd Y_p=MatrixXd::Zero(X.rows(),1);
	int n = (int) K_tr.rows();  
	MatrixXd Unitm = MatrixXd::Identity(n, n); 
	
	MatrixXd Inv_G = (K_tr + n*lamda_*Unitm).inverse();
	VectorXd Unitv = VectorXd::Ones(n);
	double Y_u = Y_tr.mean();
	MatrixXd Y_t = Y_tr.array() - Y_u;
	for (int i = 0; i < X.rows(); ++i)
	{
		//prediction for single item
		VectorXd k_g = VectorXd::Zero(n);
		k_g = (-1 * (((Unitv*X.row(i)) - X_tr).rowwise().squaredNorm() / (2 * sigma_*sigma_))).array().exp();
		Y_p(i, 0) = Y_u + (Y_t.transpose()*Inv_G*k_g)(0, 0);
	}
	return Y_p;
}
 