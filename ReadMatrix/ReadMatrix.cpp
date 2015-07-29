#include "ReadMatrix.h"
#include <iostream>
#include <fstream>
#include <time.h>
#include <iomanip>
#define LINE cout<<string(65,'-')<<endl;
using namespace std;
using namespace Eigen;

MatrixXd ReadMatrix::read(const string& filename, const string& delimiter , int rows , int cols )
{
	LINE; LINE;
	cout << "Reading data from: " << filename << endl;
	cout << "Using delimiter [ " << delimiter << " ]" << endl;
	cout << "Reading......  ";
	fstream in(filename);
	string dataLine;
	string lineBuf;
	int col = cols, row = rows;

	if (rows == 0)
	{
		//get how many lines 
		if (getline(in, dataLine)) { ++row; }
		while (getline(in, lineBuf)) { ++row; }

		//get how many colums 
		int start = dataLine.find_first_not_of(delimiter, 0);
		while (start < (int)dataLine.size())
		{
			auto cur = dataLine.find_first_of(delimiter, start);
			if (cur == string::npos) break;
			double field = stod(dataLine.substr(start, cur - start));
			col++;
			start = dataLine.find_first_not_of(delimiter, cur);
		}
		if (start != string::npos){
			double lastField = stod(dataLine.substr(start));
			col++;
		}

		in.clear();
		in.seekg(0, ios::beg);
	}

	//read data
	MatrixXd data(row, col);
	int totalRow = row;
	col = 0, row = 0;
	while (getline(in, dataLine)) // get each line
	{
		if (row == totalRow / 3) cout << " 30 % ";
		if (row == 2 * totalRow / 3) cout << " 60 % ";
		// extract fields
		col = 0;
		int start = dataLine.find_first_not_of(delimiter, 0);
		while (start< (int)dataLine.size())
		{
			auto cur = dataLine.find_first_of(delimiter, start);
			if (cur == string::npos) break;
			double field = stod(dataLine.substr(start, cur - start));
			data(row, col++) = field;
			start = dataLine.find_first_not_of(delimiter, cur);
		}
		if (start != string::npos){
			double lastField = stod(dataLine.substr(start));
			data(row, col++) = lastField;
		}
		++row;
	}
	in.close();
	cout << endl;
	return data;
}
 