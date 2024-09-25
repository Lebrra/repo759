#include <iostream>
#include <chrono>

#include "convolve.cpp"
#include "convolve.h"

using namespace std;

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
	cout << "time to process:\t" << (timePassed.count() / 1000) << " milliseconds\n";
	cout << "first element:  \t" << g[0] << endl;
	cout << "last element:   \t" << g[n*n - 1] << endl;
	
	
	free(randomF);
	free(randomW);
	free(g);
}

