#include <cstddef>
using namespace std;

#ifndef MATMUL_H

void mmul(const float* A, const float* B, float* C, size_t n) {
	for (int i = 0; i < n; i++) {
		for (int k = 0; k < n; k++) {
			for (int j = 0; j < n; j++) {
				C[i * n + j] += A[i * n + k] * B[k * n + j];
			}
		}
	}
}

#endif