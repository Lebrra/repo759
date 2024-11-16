#include <cuda.h>
#include <stdio.h>

#ifndef MATMUL_CUH

// Computes the matrix product of A and B, storing the result in C.
// Each thread should compute _one_ element of output.
// Does not use shared memory for this problem.
//
// A, B, and C are row major representations of nxn matrices in device memory.
//
// Assumptions:
// - 1D kernel configuration
__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n){
    // threadIdx = i
    // blockIdx = j
    // need to iterate k

    int iIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (iIndex >= n*n) return;

    for (int k = 0; k < n; k++){
        int jIndex = (iIndex / n) * n + k;
        int kIndex = k * n + (iIndex % n);
        if (k == 0) printf("INDEX 0 -> index = %d | j = %d | k = %d\n", iIndex, jIndex, kIndex);
        if (k == 1) printf("INDEX 1 -> index = %d | j = %d | k = %d\n", iIndex, jIndex, kIndex);
        C[iIndex] += A[jIndex] * B[kIndex];
    }
}

// Makes one call to matmul_kernel with threads_per_block threads per block.
// You can consider following the kernel call with cudaDeviceSynchronize (but if you use 
// cudaEventSynchronize to time it, that call serves the same purpose as cudaDeviceSynchronize).
void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block){
    int blocks = (n*n + threads_per_block - 1) / threads_per_block;
    matmul_kernel<<<blocks, threads_per_block>>>(A, B, C, n);
    cudaDeviceSynchronize();
}

#endif