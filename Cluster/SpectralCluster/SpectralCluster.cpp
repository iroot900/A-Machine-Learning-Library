#include "SpectralCluster.h"
#include "Eigen\Dense"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

using namespace std;
using namespace Eigen;
#define LINE cout<<string(65,'-')<<endl;

void SP::train(const MatrixXd& data, int k)
{ 
	LINE;
	cout << "Data Dimension : " << data.rows() << " X " << data.cols() << endl;
	LINE;
	cout << endl;

	int row = data.rows(); int col = data.cols();
	LINE;

	//Similarity Graph  (weight matrix-- k nearest neighbor)
	MatrixXd W = MatrixXd::Zero(row, row);
	//int k = 70; // Model selection can be tricky!
	for (int i = 0; i < row; ++i)
	{
		std::vector<pair<double, int>> dist;
		for (int j = 0; j < row; ++j)
		{
			dist.emplace_back((data.row(i) - data.row(j)).norm(), j);
		}
		std::sort(begin(dist), end(dist), [](pair<double, int> left, pair<double, int> right) {return left.first < right.first; });
		for (int t = 0; t < k; ++t) { W(i, dist[t].second) = 1; }
	}
	cout << "Build similarity Graph using k nearest neighbor... " << endl << endl;

	//Laplacian Matrix 
	MatrixXd L = MatrixXd::Zero(row, row);
	for (int i = 0; i < row; ++i)
	{
		for (int j = 0; j < row; ++j)
		{
			if (i != j) L(i, j) = -W(i, j);
			else L(i, j) = 10 - W(i, j);
		}
	}

	//Spectral decomposition of L
	SelfAdjointEigenSolver<MatrixXd> eg(L);
	VectorXd egValue = eg.eigenvalues();
	K = 1;
	for (; K < row; ++K)
	{
		if (abs(egValue(K) - egValue(K - 1)) / abs(egValue(K - 1)) >0.55) break;
		//again, Model selection can be tricky. choose the right threshhold for K means
	}
	++K;
	//cout << egValue.transpose()<<endl;
	cout << "Spectral decomposition of Laplacian Matrix..." << endl << endl;
	//Transformed data (Nullspace of L)
	MatrixXd data_(row, K);
	{
		MatrixXd egVector = eg.eigenvectors();
		for (int i = 0; i < K; ++i) { data_.col(i) = egVector.col(i); }
	}

	//K-means
	int len = row / K;
	MatrixXd ku = MatrixXd::Zero(K, col);
	for (int i = 0; i < K; ++i) { ku.row(i) = data_.row(i*len); }

	int count = 0;
	double last = INT_MAX;
	double cur = 0;
	std::vector<char> label(row);
	while (1)
	{
		MatrixXd ksum = MatrixXd::Zero(K, col);
		VectorXd kcount = VectorXd::Zero(K);
		cur = 0;
		for (int i = 0; i < row; ++i)
		{
			int kth = 0; double mmin = (data_.row(i) - ku.row(0)).norm(); label[i] = 'A';
			for (int j = 1; j < K; ++j)
			{
				if ((data_.row(i) - ku.row(j)).norm() < mmin) { mmin = (data_.row(i) - ku.row(j)).norm(); kth = j; label[i] = 'A' + j; }
			}
			cur += mmin;
			kcount(kth) += 1;
			ksum.row(kth) += data_.row(i);
		}
		if (abs(cur - last) < 1) break; last = cur;
		for (int i = 0; i < K; ++i) { ku.row(i) = ksum.row(i) / kcount(i); }
	}
	cout << "K Means cluster on Nullspace of Laplacian matrix..." << endl << endl;
	LINE;
	LINE;

	cout << "Result: " << endl << endl;
	cout << K << " clusters been chosen!" << endl << endl;
	LINE;
	cout << "Clustered Label: " << endl << endl;
	for (auto labl : label) cout << labl << " "; cout << endl << endl;
}
 