#include "ReadMatrix.h"
#include "KernelRegression.h"
#include "Eigen\Dense"
#include <iostream>
#include <iomanip>
using namespace std;
using namespace Eigen;
#define LINE cout<<string(65,'-')<<endl;

int main()
{
	ReadMatrix dataReader;
	string filename = "bodyfat_data";
	string delimiter = ",	"; 
	MatrixXd data = dataReader.read(filename, delimiter);
	 
	KR kr;
	kr.train(data);

	//predict on first 20 X;
	LINE;
	cout << "Prediction of y on first 20 x: " << endl << endl;
	cout << kr.predict(data.block(0, 0, 10, data.cols() - 1)).transpose() << endl << endl;
	cout << (data.block(0, data.cols() - 1, 10, 1)).transpose() << endl;
	LINE;

}
 