#include "KernelDensityEstimation.h"
#include "Eigen\Dense"
#include <iostream>
#include <iomanip>
using namespace std;
using namespace Eigen;
#define LINE cout<<string(65,'-')<<endl;

void KDE::train(const MatrixXd& data)
{

	LINE;
	cout << "Data Dimension : " << data.rows() << " X " << data.cols() << endl;
	LINE;
	cout << endl;

	int row = (int)data.rows();

	cout << "Kernel (Gaussian) Density Estimation and Model selection (Sigma) :" << endl;
	cout << endl;

	double minIntSqrError = DBL_MAX;
	double bestSigmas = -1.0;

	//Model selection: sigma's range [0.5, 2.5]
	//This have to be very delicate range, since we're using estimation already for the fmean.
	VectorXd sigmas(100); sigmas(0) = 0.5;
	for (int i = 1; i < 100; ++i) sigmas(i) = sigmas(i - 1) + 0.02;

	//first two terms of integrated square error
	for (int i = 0; i < 100; ++i)
	{
		double sigma = sigmas(i);
		double fmean = .0;
		double fsquare = .0;
		for (int j = 0; j < row; ++j)
		{
			VectorXd xx = data.row(j);
			double fx = .0;
			for (int i = 0; i < row; ++i)
			{
				fsquare += Gaussian(xx, std::sqrt(2)*sigma, data.row(i));
				if (i != j)
					fx += Gaussian(xx, sigma, data.row(i));
			}
			fx /= (row - 1);
			fmean += fx;
		}
		fmean = fmean / row;
		double ferror = (1.0 / (row*row))*fsquare - (2.0)*fmean;
		if (ferror < minIntSqrError) { minIntSqrError = ferror; bestSigmas = sigmas(i); }
		cout << "Sigma = " << setw(4) << sigmas(i) << " , Integrated Square Error (first two) = " << setw(10) << ferror << endl;
	}

	LINE;
	cout << endl;
	LINE;
	cout << "Result:" << endl << endl;
	cout << "Gaussian Kernel used:" << endl;
	cout << "The best value for bandWidth (sigma) : " << bestSigmas << endl;
	sigma = bestSigmas; 
	LINE;
}

MatrixXd KDE::estimateDensity(const MatrixXd& data)
{  
	MatrixXd density(data.rows(),1);

	for (int i = 0; i < data.rows() ;++i)
	{ 
		double fx=0;
		for (int j = 0; j < data.rows(); ++j)
		{
			fx += Gaussian(data.row(i), sigma, data.row(j));
		}
		fx /= data.rows();
		density(i,0) = fx;
	}
	return density;
}
