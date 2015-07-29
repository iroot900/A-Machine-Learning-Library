#ifndef KMEANS_H
#define KMEANS_H
#include "Eigen\Dense"
#include <string>
using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::VectorXd;

class KM
{
public:
	void train(const MatrixXd& data);
	MatrixXd clusterLabel(const MatrixXd& X);

private: 
	//trained parameters
	MatrixXd ku; //k center vectors;
	int K;//number of center chosen
};
#endif