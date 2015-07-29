#include "ReadMatrix.h"
#include "LeastAbsoluteRegression.h"
#include "Eigen\Dense"
#include <iostream>
using namespace std;
using namespace Eigen;
#define LINE cout<<string(65,'-')<<endl;

int main()
{
	ReadMatrix dataReader;
	string filename = "dataset";
	string delimiter = ",	"; 
	MatrixXd data = dataReader.read(filename, delimiter);
	
	LINE;
	LAR lar;
	lar.train(data);

	//predict on first 20 X;
	LINE;
	cout << "Prediction of y on first 20 x: " << endl << endl;
	cout << lar.predict(data.block(0, 0, 20, data.cols() - 1)).transpose() << endl<<endl;
	cout << (data.block(0, data.cols() - 1, 20, 1)).transpose() << endl;
	LINE;
}
 