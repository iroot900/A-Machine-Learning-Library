#include "LogisticRegression.h"
#include "Eigen\Dense"
#include <iostream>
using namespace std;
using namespace Eigen;
#define LINE cout<<string(65,'-')<<endl;

void LR::train(const MatrixXd& data, double trainRatio, double lamda, double alpha)
{
	//Training data and testing data;
	int breakPoint = int(data.rows() * trainRatio), col = data.cols(), row = data.rows();
	//training
	MatrixXd offsetTr = MatrixXd::Ones(breakPoint, 1);
	MatrixXd Xtr_(breakPoint, col);
	Xtr_ << data.block(0, 0, breakPoint, col - 1), offsetTr;
	MatrixXd Ytr = data.block(0, col - 1, breakPoint, 1).array(); 
	if (!zeroOne) Ytr = (Ytr.array() + 1) / 2;

	//testing
	MatrixXd offsetTs = MatrixXd::Ones(row - breakPoint, 1);
	MatrixXd Xts_(row - breakPoint, col);
	Xts_ << data.block(breakPoint, 0, row - breakPoint, col - 1), offsetTs;
	MatrixXd Yts = data.block(breakPoint, col - 1, row - breakPoint, 1).array(); 
	if (!zeroOne) Yts = (Yts.array() + 1) / 2;

	// initialize the parameter
	MatrixXd theta_old = MatrixXd::Ones(col, 1);
	MatrixXd theta_new = MatrixXd::Zero(col, 1);

	//gradient method: 
	int i = 0;
	LINE;
	cout << "Training : " << endl <<endl;
	while ((theta_old - theta_new).norm() > 0.025)
	{
		cout << "Training iteration: " << ++i << endl;
		theta_old = theta_new;
		ArrayXd  exAry = (Xtr_*theta_old).array().exp();
		MatrixXd gradient_ = Xtr_.transpose() *(((exAry / (1 + exAry)) - Ytr.array()).matrix());
		MatrixXd gradient = gradient_ + 2 * lamda* theta_old;
		theta_new = theta_old - alpha* gradient;

		double objVal = ((1 + (1 / exAry)).log()).sum() + ((Xtr_*theta_new).array()*(1 - Ytr.array())).sum() + lamda*(theta_new.transpose()*theta_new)(0);
		cout << "Value of objective function: " << objVal << endl;
	}

	theta = theta_new;
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

MatrixXd LR::predict(const MatrixXd& X)
{
	MatrixXd label;
	if (X.cols() != theta.size())
	{
		MatrixXd X_(X.rows(), X.cols());
		X_ << X, MatrixXd::Ones(X.rows(),1);
		label = (X_*theta).array();
	}
	else label=(X*theta).array();
	for (int i = 0; i < label.size(); i++)
	{
		if (label(i)>0) label(i) = 1;
		else label(i) = 0;
	} 
	return label;
}
