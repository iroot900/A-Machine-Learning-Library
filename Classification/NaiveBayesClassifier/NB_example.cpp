#include "ReadMatrix.h"
#include "NaiveBayesClassifier.h"
#include "Eigen\Dense"
#include <iostream>
using namespace std;
using namespace Eigen;
#define LINE cout<<string(65,'-')<<endl;

int main()
{
	// detect nuclear particle type
	ReadMatrix dataReader;
	string filename = "spamdata"; 
	string delimiter = ", ";
	MatrixXd data= dataReader.read(filename, delimiter);

	LINE;  
	NB nb;
	nb.train(data);
	LINE;

	cout << "Prediction of the first five labels: " << endl;
	MatrixXd X_ = data.block(0, 0, 5, data.cols() - 1);
	cout << nb.predict(X_).transpose() << endl;
	LINE; LINE;
}