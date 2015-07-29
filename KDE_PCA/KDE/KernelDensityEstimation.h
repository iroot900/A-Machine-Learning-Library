#ifndef KERNEL_DENSITY_ESTIMATION_H
#define KERNEL_DENSITY_ESTIMATION_H
#include "Eigen\Dense"
#include <string>
using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::VectorXd;

class KDE
{
public:
	void train(const MatrixXd& data);
	MatrixXd estimateDensity(const MatrixXd& X); 

private:

	double Gaussian(VectorXd x, double sigma, VectorXd u)
	{
		int d = x.size();

		return pow(2 * 3.14, -d / 2)*pow(sigma, -d)*exp((-1 / (2 * sigma*sigma))* ((x - u).squaredNorm()));
	}
	//trained parameters
	double sigma; 
};
#endif