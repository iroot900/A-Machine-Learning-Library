#include "LinearDiscriminantAnalysis.h"
#include "Eigen\Dense"
#include <iostream>
using namespace std;
using namespace Eigen;
#define LINE cout<<string(65,'-')<<endl;

void LDA::train(const MatrixXd& data, double trainRatio)
{
	//Training data and testing data;
	int breakPoint = int(data.rows() * trainRatio), col = data.cols(), row = data.rows();

	//training
	MatrixXd Xtr_ = data.block(0, 0, breakPoint, col - 1);
	MatrixXd Ytr = data.block(0, col - 1, breakPoint, 1).array();
	if (!zeroOne) Ytr = (Ytr.array() + 1) / 2;

	//testing
	MatrixXd Xts_ = data.block(breakPoint, 0, row - breakPoint, col - 1);
	MatrixXd Yts = data.block(breakPoint, col - 1, row - breakPoint, 1).array();
	if (!zeroOne) Yts = (Yts.array() + 1) / 2;

	cout << endl<< "Estimating parameters... " << endl << endl;
	//Linear Discriminant  Analysis: 
	int N_0 = (Ytr.array() == 0).count();
	int N_1 = (Ytr.array() == 1).count();

	//estimator for pi_k
	pi_0 = (double)N_0 / Ytr.size();
	pi_1 = (double)N_1 / Ytr.size();

	//estimator for u_k
	u_0 = MatrixXd::Zero(1, Xtr_.cols());
	u_1 = MatrixXd::Zero(1, Xtr_.cols());
	for (int i = 0; i < Xtr_.rows(); ++i)
	{
		if (Ytr(i) == 0) u_0 = u_0 + Xtr_.row(i);
		else u_1 = u_1 + Xtr_.row(i);
	}

	u_0 = u_0 / N_0;
	u_1 = u_1 / N_1;

	//estimator for covariance matrix 
	covariance = MatrixXd::Zero(Xtr_.cols(), Xtr_.cols());

	for (int i = 0; i < Xtr_.rows(); ++i)
	{
		Xtr_.row(i) = (Xtr_.row(i) - u_0);
	}
	covariance = (Xtr_.transpose()*Xtr_) / Xtr_.rows();
	 
	//classifying training set... 
	labelTr = predict(Xtr_);

	//classifying testing set...
	labelTs = predict(Xts_); 

	// Error on training and test set
	int corretLableTr = ((labelTr.array() - Ytr.array()).abs() < 0.1).count();
	int corretLableTs = ((labelTs.array() - Yts.array()).abs() < 0.1).count();
	ErrorTr = (breakPoint - corretLableTr) / (double)breakPoint;
	ErrorTs = (row - breakPoint - corretLableTs) / (double)(row - breakPoint);

	LINE; LINE;
	cout << "Result: " << endl;
	cout << "Error On Training Set(" << breakPoint << " x " << col - 1 << ") : " << ErrorTr * 100 << "%" << endl;
	cout << "Error On Testing  Set(" << row - breakPoint << " x " << col - 1 << ") : " << ErrorTs * 100 << "%" << endl;
	LINE;
}

MatrixXd LDA::predict(const MatrixXd& X)
{
	ArrayXd label = ArrayXd::Zero(X.rows(), 1);
	MatrixXd inverse = covariance.inverse();
	double norm = covariance.norm();

	for (int i = 0; i < X.rows(); ++i)
	{
		double a = (-(1 / 2.0)*((X.row(i) - u_0) *inverse*(X.row(i) - u_0).transpose()))(0, 0);
		//double density_0 = pi_0*(std::pow(2 * 3.14, -Xtr_.cols() / 2))*std::pow(norm, -1 / 2)*std::exp(a);
		double density_0 = pi_0*std::exp(a);
		a = (-(1 / 2.0)*((X.row(i) - u_1) *inverse*(X.row(i) - u_1).transpose()))(0, 0);
		//double density_1 = pi_1*(std::pow(2 * 3.14, -Xtr_.cols() / 2))*std::pow(norm, -1 / 2)*std::exp(a);
		double density_1 = pi_1*std::exp(a);

		if (density_0 > density_1) label(i) = 0;
		else label(i) = 1;
	}
	return label;
}
