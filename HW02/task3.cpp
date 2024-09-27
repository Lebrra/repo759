#include <iostream>
#include <chrono>
#include <vector>

#include "matmul.cpp"
#include "matmul.h"

using namespace std;

void empty(double* arr, int length) {
	if (arr) {
		for (int i = 0; i < length; i++) {
			arr[i] = 0;
		}
	}
}

void randFill(double* arr, vector<double>& vect, int length) {
	vect.clear();

	if (arr) {
		for (int i = 0; i < length; i++) {
			double r = static_cast <float> (rand() / static_cast <float> (RAND_MAX / 2)) - 1;
			arr[i] = r;
			vect.push_back(r);
		}
	}
}

void printArray(double* arr, int n) {
	if (arr) {
		cout << "Matrix:\n";
		for (int i = 0; i < n * n; i++) {
			cout << arr[i] << " ";
			if (i % n == n - 1) cout << endl;
		}
		cout << endl;
	}
}

int main(int argc, char* argv[])
{
	int n = 200;

	// seeding with chrono so I don't have to include the ctime library:
	srand(chrono::system_clock::now().time_since_epoch().count());

	// create data:
	double* a = (double*)malloc(sizeof(double) * n * n);
	double* b = (double*)malloc(sizeof(double) * n * n);
	double* c = (double*)malloc(sizeof(double) * n * n);

	vector<double> aVect;
	vector<double> bVect;
	
	randFill(a, aVect, n * n);
	randFill(b, bVect, n * n);

	matmul mat;

	for (int i = 0; i < 4; i++) {
		empty(c, n * n);

		auto start = chrono::steady_clock::now();

		switch (i) {
			case 0: 
				mat.mmul1(a, b, c, n);
				break;
			case 1:
				mat.mmul2(a, b, c, n);
				break;
			case 2: 
				mat.mmul3(a, b, c, n);
				break;
			case 3:
				mat.mmul4(aVect, bVect, c, n);
				break;
		}

		auto end = chrono::steady_clock::now();
		auto timePassed = chrono::duration_cast<std::chrono::microseconds>(end - start);

		cout << "matmul.mmul" << (i + 1) << "()" << endl;
		cout << "time to process:\t" << (timePassed.count() / 1000) << " milliseconds\n";
		if (c) cout << "last element:\t\t" << c[n * n - 1] << endl;
		cout << endl;
		//printArray(c, n);
	}

	free(a);
	free(b);
	free(c);

	aVect = vector<double>();
	bVect = vector<double>();
}