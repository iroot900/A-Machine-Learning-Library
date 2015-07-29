#include "PrincipalComponentAnalysis.h"
#include "Eigen\Dense"
#include <iostream>
using namespace std;
using namespace Eigen;
#define LINE cout<<string(65,'-')<<endl;

void PCA::train(const MatrixXd& data_)
{
	LINE;
	cout << "Data Dimension : " << data.rows() << " X " << data.cols() << endl;
	LINE;
	cout << endl;

	MatrixXd data = data_;
	//mean vector
	MatrixXd u = data.colwise().mean();

	//covariance matrix
	for (int i = 0; i < data.rows(); ++i)
	{
		data.row(i) = data.row(i) - u;
	}
	MatrixXd cov = (data.transpose()*data) / data.rows();
	cout <<endl<< "Spectral decomposation :" << endl;
	cout << "   covariance matrix " << cov.cols() << " X " << cov.cols() << "  " << endl;
	SelfAdjointEigenSolver<MatrixXd> A(cov);
	eigenVectors = A.eigenvectors();
	eigenValues = A.eigenvalues();
	principalComps = data*eigenVectors;

	//95% percent variance
	double variance = 0;
	double totalVariance = eigenValues.sum();
	int noEig = eigenValues.size();
	bool P95 = false;
	int N95 = 0;
	int N99 = 0;

	for (int i = 0; i < noEig; ++i)
	{
		variance += eigenValues(noEig - 1 - i);
		if ((variance / totalVariance) < 0.95&&!P95) { N95 = i + 1; }
		else P95 = true;

		if ((variance / totalVariance) < 0.99) { N99 = i + 1; }
		else break;
	}

	LINE;
	cout << "Result: " << endl << endl;
	cout << "95% varaince achieved with " << N95 << " principal components," << endl;
	cout << "    Dimension from" << noEig << " to " << N95 << ", a " << ((double)(noEig - N95) / noEig) * 100 << "% reduction." << endl;

	cout << endl << endl;
	cout << "99% varaince achieved with " << N99 << " principal components," << endl;
	cout << "    Dimension from" << noEig << " to " << N99 << ", a " << ((double)(noEig - N99) / noEig) * 100 << "% reduction." << endl;
	LINE;
	cout << endl;
}

MatrixXd PCA::to_principal (const MatrixXd& data_)
{
	MatrixXd data = data_;
	MatrixXd u = data.colwise().mean();

	//covariance matrix
	for (int i = 0; i < data.rows(); ++i)
	{
		data.row(i) = data.row(i) - u;
	}
	return data*eigenVectors;
}
