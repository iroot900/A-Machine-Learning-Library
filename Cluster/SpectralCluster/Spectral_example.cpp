#include "ReadMatrix.h"
#include "SpectralCluster.h"
#include "Eigen\Dense"
#include <iostream>
using namespace std;
using namespace Eigen;
#define LINE cout<<string(65,'-')<<endl;

int main()
{
	// detect nuclear particle type
	ReadMatrix dataReader;
	string filename = "pathbased.txt";
	string delimiter = ",	"; 
	MatrixXd data = dataReader.read(filename, delimiter);

	LINE;
	SP sp;
	sp.train(data.block(0, 0, data.rows(), data.cols() - 1)); // this dataset happen to have 
	LINE; 
	LINE;
	cout << "Clusters label on orignal data: " << endl;
	cout << data.col(data.cols() - 1).transpose() << endl;
	LINE;  
}
 