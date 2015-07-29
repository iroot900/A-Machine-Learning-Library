#include "NaiveBayesClassifier.h"
#include "Eigen\Dense"
#include <iostream>
using namespace std;
using namespace Eigen;
#define LINE cout<<string(65,'-')<<endl;

void NB::train(const MatrixXd& data, double trainRatio)
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

	//Naive Bayes method: 
	int N_0 = (Ytr.array() == 0).count();
	int N_1 = (Ytr.array() == 1).count();

	//estimator for pi_k
	pi_0 = (double)N_0 / Ytr.size();
	pi_1 = (double)N_1 / Ytr.size();

	VectorXd meanV = Xtr_.colwise().mean();
	g_0 = MatrixXd::Ones(2, Xtr_.cols()); // divide by n_0
	g_1 = MatrixXd::Ones(2, Xtr_.cols()); // divide by n_1

	// quantify each feature into two category and count frequency 
	for (int i = 0; i < Xtr_.rows(); ++i)
	{
		for (int j = 0; j < Xtr_.cols(); ++j)
		{
			if (Ytr(i) == 0) // Y is 0   
			{
				if (Xtr_(i, j) < meanV(j))  { g_0(0, j) += 1; Xtr_(i, j) = 0; }
				else { g_0(1, j) += 1; Xtr_(i, j) = 1; }
			}
			else // Y is 1 
			{
				if (Xtr_(i, j) < meanV(j))  { g_1(0, j) += 1; Xtr_(i, j) = 0; }
				else { g_1(1, j) += 1; Xtr_(i, j) = 1; }
			}
		}
	}
	 
	//estimator for g_kl(j)
	g_0 = g_0.array() / (double)N_0;
	g_1 = g_1.array() / (double)N_1;

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

MatrixXd NB::predict(const MatrixXd& X)
{
	MatrixXd label(X.rows(), 1);

	for (int i = 0; i < X.rows(); ++i)
	{
		double density_0 = pi_0;
		double density_1 = pi_1;
		for (int j = 0; j < X.cols(); ++j)
		{
			if (X(i, j) == 0)
			{
				density_0 *= g_0(0, j);
				density_1 *= g_1(0, j);
			}
			else
			{
				density_0 *= g_0(1, j);
				density_1 *= g_1(1, j);
			}
		}
		if (density_0 > density_1) { label(i) = 0; }
		else label(i) = 1;
	}
	return label;
}
