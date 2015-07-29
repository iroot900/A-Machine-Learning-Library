#include "ReadMatrix.h"
#include "KernelDensityEstimation.h"
#include "Eigen\Dense"
#include <iostream>
using namespace std;
using namespace Eigen;
#define LINE cout<<string(65,'-')<<endl;

int main()
{
	// detect nuclear particle type
	ReadMatrix dataReader;
	string filename = "examScores.txt";
	string delimiter = ", ";
	MatrixXd data = dataReader.read(filename, delimiter);

	LINE;
	KDE kde;
	kde.train(data);
	LINE;

	cout << "Density on original points(vectors): " << endl<<endl;
	MatrixXd density=kde.estimateDensity(data); 
	cout << density.transpose() << endl<<endl;
	LINE;  
}
 