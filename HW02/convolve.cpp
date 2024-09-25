#include <iostream>
using namespace std;

#ifndef CONVOLVE_H
#define CONVOLVE_H

class convolve {
public:

	// 1D methods:
	float* doConvolve(float* f, float* w, int n, int m) {	// f = [n x n] w = [m x m]
		float* g;
		g = (float*)malloc(sizeof(float) * n * n);

		if (g) {
			for (int i = 0; i < n * n; i++) {
				int x = i / n;
				int y = i % n;

				g[i] = getValueAtXY(x, y, f, w, n, m);
			}
		}

		return g;
	}

	float getValueAtXY(int x, int y, float* f, float* w, int n, int m) {
		// m is assumed to be positive
		int mHalf = (m - 1) / 2;
		float sum = 0;

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

	// 2D overloads:
	float** doConvolve(float** f, float** w, int n, int m) {	// f = [n x n] w = [m x m]
		float** g;
		g = (float**)malloc(sizeof(float*) * n);

		if (g) {
			for (int x = 0; x < n; x++) {
				g[x] = (float*)malloc(sizeof(float) * n);

				if (g[x]) {
					for (int y = 0; y < n; y++) {
						g[x][y] = getValueAtXY(x, y, f, w, n, m);
					}
				}
			}
		}

		return g;
	}

	float getValueAtXY(int x, int y, float** f, float** w, int n, int m) {
		// m is assumed to be positive
		int mHalf = (m - 1) / 2;
		float sum = 0;

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < m; j++) {
				float wVal = w[i][j];
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
				else fVal = f[fi][fj];

				sum += (wVal * fVal);
			}
		}

		return sum;
	}
};

#endif