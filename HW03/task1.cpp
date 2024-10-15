// compile with: /openmp or -fopenmp

#include <iostream>
#include <chrono>

#include "matmul.h"
#include "omp.h"

using namespace std;

void empty(float* arr, int length) {
	if (arr) {
		for (int i = 0; i < length; i++) {
			arr[i] = 0;
		}
	}
}

void randFill(float* arr, int length) {
	if (arr) {
		for (int i = 0; i < length; i++) {
			float r = static_cast <float> (rand() / static_cast <float> (RAND_MAX / 2)) - 1;
			arr[i] = r;
		}
	}
}

int main(int argc, char* argv[])
{
	int n = atoi(argv[1]);
	int t = atoi(argv[2]);

	// setting thread count this way works but declaring it below isn't for some reason...
	omp_set_num_threads(t);

#pragma omp.h parallel num_threads(t)
	{
		// seeding with chrono so I don't have to include the ctime library:
		srand(chrono::system_clock::now().time_since_epoch().count());
	
		// create data:
		float* a = (float*)malloc(sizeof(float) * n * n);
		float* b = (float*)malloc(sizeof(float) * n * n);
		float* c = (float*)malloc(sizeof(float) * n * n);
	
		randFill(a, n * n);
		randFill(b, n * n);
		empty(c, n * n);
	
		auto start = chrono::steady_clock::now();
	
		mmul(a, b, c, n);
	
		auto end = chrono::steady_clock::now();
		auto timePassed = chrono::duration_cast<std::chrono::microseconds>(end - start);
	
		cout << "Results for n=" << n << " & t=" << t << endl;
		cout << "time to process:\t" << (timePassed.count() / 1000) << " milliseconds\n";
		if (c) 
		{
			cout << "first element:\t\t" << c[0] << endl;
			cout << "last element:\t\t" << c[n * n - 1] << endl;
		}
		cout << endl;
	
		free(a);
		free(b);
		free(c);
	}
	return 0;
}