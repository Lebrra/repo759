#include <iostream>
#include "scan.h"

using namespace std;

void scan(const float* arr, float* output, std::size_t n) {
	if (output) {	// added null check here to remove warning C6011
		output[0] = arr[0];
		for (int i = 1; i < n; i++) {
			output[i] = output[i - 1] + arr[i];
		}
	}
}

