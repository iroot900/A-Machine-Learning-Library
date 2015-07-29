#include "ReadMatrix.h"
#include "LogisticRegression.h"
#include "Eigen\Dense"
#include <iostream>
using namespace std;
using namespace Eigen;
#define LINE cout<<string(65,'-')<<endl;

int main()
{
	// classify hand written digit data 

	ReadMatrix dataReader;
	string filename = "digit_data"; 
	string delimiter = ", ";
	MatrixXd data= dataReader.read(filename, delimiter);

	LINE;  
	LR lr(false); //default true as 0/1 input,  set false for -1/1 input;
	lr.train(data); 
	LINE;

	cout << "Prediction of the first five labels: " << endl;
	MatrixXd X_ = data.block(0, 0, 5, data.cols() - 1);
	cout << lr.predict(X_).transpose()<<endl;
	LINE; LINE;
}