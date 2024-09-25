#include <iostream>
#include <chrono>

#include "convolve.cpp"
#include "convolve.h"

using namespace std;

void tester2D() {
	int n = 4;
	int m = 3;

	float** f;
	f = (float**)malloc(sizeof(float*) * 4);
	f[0] = (float*)malloc(sizeof(float) * 4);
	f[0][0] = 1;
	f[0][1] = 3;
	f[0][2] = 4;
	f[0][3] = 8;
	f[1] = (float*)malloc(sizeof(float) * 4);
	f[1][0] = 6;
	f[1][1] = 5;
	f[1][2] = 2;
	f[1][3] = 4;
	f[2] = (float*)malloc(sizeof(float) * 4);
	f[2][0] = 3;
	f[2][1] = 4;
	f[2][2] = 6;
	f[2][3] = 8;
	f[3] = (float*)malloc(sizeof(float) * 4);
	f[3][0] = 1;
	f[3][1] = 4;
	f[3][2] = 5;
	f[3][3] = 2;

	cout << "F:\n";
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cout << f[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	float** w;
	w = (float**)malloc(sizeof(float*) * 3);
	w[0] = (float*)malloc(sizeof(float) * 3);
	w[0][0] = 0;
	w[0][1] = 0;
	w[0][2] = 1;
	w[1] = (float*)malloc(sizeof(float) * 3);
	w[1][0] = 0;
	w[1][1] = 1;
	w[1][2] = 0;
	w[2] = (float*)malloc(sizeof(float) * 3);
	w[2][0] = 1;
	w[2][1] = 0;
	w[2][2] = 0;

	cout << "W:\n";
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			cout << w[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	convolve c;
	float** g = c.doConvolve(f, w, n, m);
	
	cout << "G:\n";
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cout << g[i][j] << " ";
		}
		cout << endl;
	}

	delete g, f, w;
}

void tester1D() {
	int n = 4;
	int m = 3;

	float fVals[16] = { 1, 3, 4, 8, 6, 5, 2, 4, 3, 4, 6, 8, 1, 4, 5, 2 };
	float wVals[9] = {0, 0, 1, 0, 1, 0, 1, 0, 0};

	float* f;
	//f = (float*)malloc(sizeof(float) * 4 * 4);
	f = fVals;

	cout << "F:\n";
	for (int i = 0; i < n * n; i++) {
		cout << f[i] << " ";
		if (i % n == n - 1) cout << endl;
	}
	cout << endl;

	float* w;
	w = wVals;

	cout << "W:\n";
	for (int i = 0; i < m * m; i++) {
		cout << w[i] << " ";
		if (i % m == m - 1) cout << endl;
	}
	cout << endl;

	convolve c;
	float* g = c.doConvolve(f, w, n, m);
	
	cout << "G:\n";
	for (int i = 0; i < n * n; i++) {
		cout << g[i] << " ";
		if (i % n == n - 1) cout << endl;
	}

	delete g, f, w;
}

int main(int argc, char* argv[])
{
	auto start = chrono::steady_clock::now();

	int n = atoi(argv[1]);
	//int n = 6;
	int m = atoi(argv[2]);
	//int m = 5;

	if (n <= 0) {
		cout << "Invalid value of 'n' !";
		return -1;
	}
	if (m <= 0 || m % 2 != 1) {
		cout << "Invalid value of 'm' !";
		return -1;
	}

	// seeding with chrono so I don't have to include the ctime library:
	srand(chrono::system_clock::now().time_since_epoch().count());


	float* randomF;
	randomF = (float*)malloc(sizeof(float) * n*n);
	if (randomF) {	// added null check here to remove warning C6011
		for (int i = 0; i < n*n; i++) {
			randomF[i] = static_cast <float> (rand() / static_cast <float> (RAND_MAX / 20)) - 10;
		}
	}

	float* randomW;
	randomW = (float*)malloc(sizeof(float) * m*m);
	if (randomW) {	// added null check here to remove warning C6011
		for (int i = 0; i < m*m; i++) {
			randomW[i] = static_cast <float> (rand() / static_cast <float> (RAND_MAX / 2)) - 1;
		}
	}

	convolve c;
	float* g = c.doConvolve(randomF, randomW, n, m);

	auto end = chrono::steady_clock::now();
	auto timePassed = chrono::duration_cast<std::chrono::microseconds>(end - start);

	cout << "Results:\n";
	cout << "time to process:\t" << timePassed.count() << " microseconds\n";
	cout << "first element:  \t" << g[0] << endl;
	cout << "last element:   \t" << g[n*n - 1] << endl;
	
	
	delete randomF, randomW, g;
}

