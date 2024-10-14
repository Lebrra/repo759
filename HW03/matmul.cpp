#include <cstddef>
#include <iostream>
#include "omp.h"
using namespace std;

#ifndef MATMUL_H

void mmul(const float* A, const float* B, float* C, size_t n) {
#pragma omp parallel for collapse(3)
	{
		for (int i = 0; i < n; i++) {
			for (int k = 0; k < n; k++) {
				for (int j = 0; j < n; j++) {
					C[i * n + j] += A[i * n + k] * B[k * n + j];
					printf("i = %d, k = %d, j = %d, threadId = %d \n", i, k, j, omp_get_thread_num());
				}
			}
		}
	}

}

#endif
