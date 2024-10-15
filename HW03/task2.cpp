#include <iostream>
#include <chrono>

#include "omp.h"
#include "convolution.h"

using namespace std;

int main(int argc, char* argv[])
{
	auto start = chrono::steady_clock::now();

	int n = atoi(argv[1]);
	int m = 3;
	int t = atoi(argv[2]);

	if (n <= 0) {
		cout << "Invalid value of 'n' !";
		return -1;
	}

	// seeding with chrono so I don't have to include the ctime library:
	srand(chrono::system_clock::now().time_since_epoch().count());

	float* randomF;
	randomF = (float*)malloc(sizeof(float) * n * n);
	if (randomF) {	// added null check here to remove warning C6011
		for (int i = 0; i < n * n; i++) {
			randomF[i] = static_cast <float> (rand() / static_cast <float> (RAND_MAX / 20)) - 10;
		}
	}

	float* randomW;
	randomW = (float*)malloc(sizeof(float) * m * m);
	if (randomW) {	// added null check here to remove warning C6011
		for (int i = 0; i < m * m; i++) {
			randomW[i] = static_cast <float> (rand() / static_cast <float> (RAND_MAX / 2)) - 1;
		}
	}

	float* g;
	g = (float*)malloc(sizeof(float) * n * n);

	// setting thread count this way works but declaring it below isn't for some reason...
	omp_set_num_threads(t);

#pragma omp parallel num_threads(t)
	{
		if (g) {
			convolve(randomF, g, n, randomW, m);

			auto end = chrono::steady_clock::now();
			auto timePassed = chrono::duration_cast<std::chrono::microseconds>(end - start);

			cout << "Results: (n=" << n << ", t=" << t << ")\n";
			cout << "time to process:\t" << (timePassed.count() / 1000) << " milliseconds\n";
			cout << "first element:  \t" << g[0] << endl;
			cout << "last element:   \t" << g[n * n - 1] << endl;

			free(g);
		}
	}
	
	free(randomF);
	free(randomW);
}

