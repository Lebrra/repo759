#include <cstddef>
#include "omp.h"
using namespace std;

#ifndef MATMUL_H

void mmul(const float* A, const float* B, float* C, size_t n) {
//#pragma omp parallel for collapse(3)
#pragma omp parallel for
	for (int i = 0; i < n; i++) {
		#pragma omp parallel for
		for (int k = 0; k < n; k++) {
			#pragma omp parallel for
			for (int j = 0; j < n; j++) {
				C[i * n + j] += A[i * n + k] * B[k * n + j];
			}
		}
	}

}

#endif
