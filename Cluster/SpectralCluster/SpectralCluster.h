#ifndef SPECTRAL_CLUSTER_H
#define SPECTRAL_CLUSTER_H
#include "Eigen\Dense"
#include <string>
using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::VectorXd;

class SP
{
public:
	void train(const MatrixXd& data, int k=70);
	MatrixXd clusterLabel(const MatrixXd& X);

private: 
	//trained parameters
	int K;//number of cluster chosen
	int k; // K nearest number for similarity graph
};
#endif