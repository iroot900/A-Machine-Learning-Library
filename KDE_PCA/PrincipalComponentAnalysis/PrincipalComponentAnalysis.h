#ifndef PRINCIPAL_COMPONENT_ANALYSIS_H
#define PRINCIPAL_COMPONENT_ANALYSIS_H
#include "Eigen\Dense"
#include <string>
using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::VectorXd;

class PCA
{
public:
	void train(const MatrixXd& data);
	MatrixXd to_principal(const MatrixXd& X); 

private:

	//trained parameters
	MatrixXd eigenVectors;
	VectorXd eigenValues;
	MatrixXd principalComps;
};
#endif