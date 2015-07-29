#ifndef  READ_MATRIX_H
#define  READ_MATRIX_H 
#include "Eigen\Dense"
#include <string>
using std::string;
using Eigen::MatrixXd;

class ReadMatrix
{
public:
	MatrixXd read(const string& filename, const string& delimiter = ",	", int rows = 0, int cols = 0);
};

#endif 