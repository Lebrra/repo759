#include <iostream>
#include <chrono>
#include <algorithm>

#include "omp.h"
#include "msort.h"

using namespace std;

int main(int argc, char* argv[])
{
	auto start = chrono::steady_clock::now();

	int n = atoi(argv[1]);
	int t = atoi(argv[2]);
	int ts = atoi(argv[3]);

	if (n <= 0) {
		cout << "Invalid value of 'n' !";
		return -1;
	}
	if (t < 1 || t > 20) {
		cout << "Invalid value of 't' !";
		return -1;
	}

	// seeding with chrono so I don't have to include the ctime library:
	srand(chrono::system_clock::now().time_since_epoch().count());

	int* arrayToSort;
	arrayToSort = (int*)malloc(sizeof(int) * n);
	if (arrayToSort) {	// added null check here to remove warning C6011
		for (int i = 0; i < n; i++) {
			arrayToSort[i] = static_cast <int> (rand() / static_cast <int> (RAND_MAX / 2000)) - 1000;
		}
	}

	// setting thread count this way works but declaring it below isn't for some reason...
	omp_set_num_threads(t);

	if (arrayToSort) {
		msort(arrayToSort, n, ts);

		auto end = chrono::steady_clock::now();
		auto timePassed = chrono::duration_cast<std::chrono::microseconds>(end - start);

		cout << "Results: (n=" << n << ", t=" << t << ", ts=" << ts << ")\n";
		cout << "time to process:\t" << (timePassed.count() / 1000) << " milliseconds\n";
		cout << "first element:  \t" << arrayToSort[0] << endl;
		cout << "last element:   \t" << arrayToSort[n - 1] << endl;
		cout << "last 3 elements:   \t" << arrayToSort[n - 3] << ",  " << arrayToSort[n - 2] << ",  " << arrayToSort[n - 1] << endl;
	}
	else {
		cout << "Error allocating array for msort!\n";
		cout << "(n=" << n << ", t=" << t << ", ts=" << ts << ")\n";
	}

	free(arrayToSort);
}