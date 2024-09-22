#include <iostream>
#include <chrono>

#include "scan.cpp"
#include "scan.h"

using namespace std;

int main(int argc, char* argv[])
{
	auto start = chrono::steady_clock::now();

	int n = atoi(argv[1]);

	if (n <= 0) {
		cout << "Invalid value of 'n' !";
		return -1;
	}

	// seeding with chrono so I don't have to include the ctime library:
	srand(chrono::system_clock::now().time_since_epoch().count());
	

	float* randFloats;
	randFloats = (float*)malloc(sizeof(float)*n);
	if (randFloats) {	// added null check here to remove warning C6011
		for (int i = 0; i < n; i++) {
			randFloats[i] = static_cast <float> (rand() / static_cast <float> (RAND_MAX / 2)) - 1;
		}
	}

	scan s;
	float* scanned = s.doScan(randFloats, n);

	auto end = chrono::steady_clock::now();
	auto timePassed = chrono::duration_cast<std::chrono::microseconds>(end - start);
	
	cout << "Results:\n";
	cout << "element count:  \t" << n << endl;
	cout << "time to process:\t" << timePassed.count() << " microseconds\n";
	cout << "first element:  \t" << scanned[0] << endl;
	cout << "last element:   \t" << scanned[n - 1] << endl << endl;
	
	free(randFloats);
	free(scanned);
}