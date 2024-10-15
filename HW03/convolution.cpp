#include <iostream>
#include "omp.h"
using namespace std;

#ifndef CONVOLVE_H

// I added this function because it made more sense to me; when I throw this all in a single method I don't get the same answers :(
float getValueAtXY(int x, int y, const float* f, const float* w, size_t n, size_t m) {
	// m is assumed to be positive
	int mHalf = (m - 1) / 2;
	float sum = 0;

	#pragma omp parallel for
	for (int all = 0; all < m * m; all++) {
		int i = all / m;
		int j = all % m;

		float wVal = w[all];
		float fVal;

		int fi = x + i - mHalf;
		int fj = y + j - mHalf;

		// if fi and fj are out of bounds:
		if ((fi < 0 || fi >= n) && (fj < 0 || fj >= n)) {
			fVal = 0;
		}
		// if fi OR fj are out of bounds:
		else if (fi < 0 || fi >= n || fj < 0 || fj >= n) {
			fVal = 1;
		}
		// fi and fj need to be mapped to n instead of m (all):
		else {
			fVal = f[fi * n + fj];
		}

		sum += (wVal * fVal);
	}

	return sum;
}

void convolve(const float* image, float* output, size_t n, const float* mask, size_t m) {
	if (output) {
		#pragma omp parallel for
		for (int i = 0; i < n * n; i++) {		// f = [n x n] w = [m x m]
			int x = i / n;
			int y = i % n;

			output[i] = getValueAtXY(x, y, image, mask, n, m);
		}
	}
}

#endif