#include "Kmeans.h"
#include "Eigen\Dense"
#include <iostream>
#include <iomanip>
using namespace std;
using namespace Eigen;
#define LINE cout<<string(65,'-')<<endl;

void KM::train(const MatrixXd& data)
{
	LINE;
	cout << "Data Dimension : " << data.rows() << " X " << data.cols() << endl;
	LINE;
	cout << endl;

	int k = 0;
	double LAST = INT_MAX;
	//parameter choosing, find the suitable K 
	cout << "Model selection for K: " << endl << endl;
	while (k < 30)
	{
		++k;
		int row = data.rows(); int col = data.cols();
		assert(k < row);
		int len = row / k;

		//K initial mean vector
		ku = MatrixXd::Zero(k, col);
		for (int i = 0; i < k; ++i) { ku.row(i) = data.row(i*len); }

		//K-means 
		int count = 0;
		double last = INT_MAX;
		double cur = 0;
		while (1)
		{
			MatrixXd ksum = MatrixXd::Zero(k, col);
			VectorXd kcount = VectorXd::Zero(k);
			cur = 0;
			for (int i = 0; i < row; ++i)
			{
				int kth = 0; double mmin = (data.row(i) - ku.row(0)).norm();
				for (int j = 1; j < k; ++j)
				{
					if ((data.row(i) - ku.row(j)).norm() < mmin) { mmin = (data.row(i) - ku.row(j)).norm(); kth = j; }
				}
				cur += mmin;
				kcount(kth) += 1;
				ksum.row(kth) += data.row(i);
			}
			if (abs(cur - last) < 1) break; last = cur;
			//update k means
			for (int i = 0; i < k; ++i) { ku.row(i) = ksum.row(i) / kcount(i); }
		}
		cout << "when K=" << setw(2) << k << " , the total variance is: " << last << endl;
		if (((LAST - last) / last) < 0.05) break;
		LAST = last;
	}
	K = k;
	LINE; LINE;
	cout << "Result: " << endl << endl;
	cout << --k << " Means(Center) been chosen! " << endl;
	LINE;
}

MatrixXd KM::clusterLabel(const MatrixXd& data)
{  
	MatrixXd center=MatrixXd::Ones(data.rows(),1);
	for (int i = 0; i < data.rows(); ++i)
	{
		int kth = 0; double mmin = (data.row(i) - ku.row(0)).norm();
		for (int j = 1; j < K; ++j)
		{
			if ((data.row(i) - ku.row(j)).norm() < mmin) { mmin = (data.row(i) - ku.row(j)).norm(); center(i,0) = j+1; }
		}
	}
	return center;
}
