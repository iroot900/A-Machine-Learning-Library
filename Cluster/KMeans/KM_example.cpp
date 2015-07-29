#include "ReadMatrix.h"
#include "Kmeans.h"
#include "Eigen\Dense"
#include <iostream>
using namespace std;
using namespace Eigen;
#define LINE cout<<string(65,'-')<<endl;

int main()
{
	// detect nuclear particle type
	ReadMatrix dataReader;
	string filename = "eigenFace";
	string delimiter = ", ";
	MatrixXd data = dataReader.read(filename, delimiter);

	LINE;
	KM km;
	km.train(data);
	LINE;

	cout << "Clusters label of original data: " << endl<<endl;
	MatrixXd clusterLabel = km.clusterLabel(data);
	cout << clusterLabel.transpose() << endl << endl;
	LINE;  
}
 