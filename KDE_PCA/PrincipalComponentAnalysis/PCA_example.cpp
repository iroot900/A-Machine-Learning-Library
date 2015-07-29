#include "ReadMatrix.h"
#include "PrincipalComponentAnalysis.h"
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
	MatrixXd data= dataReader.read(filename, delimiter);

	LINE;  
	PCA pca;
	pca.train(data);
	LINE;
	  
	pca.to_principal(data);
	LINE; LINE;
}