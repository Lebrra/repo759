#include <cuda.h>
#include <iostream> 
using namespace std; 

#ifndef MATMUL_CUH

// (the difference is types of data)

//template <typename T>
__global__ void matmul(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim){
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int c = n * block_dim * by + block_dim * bx;
    if (c >= n*n) return;

    int aStart = n * block_dim * by;
    int aEnd = aStart + n - 1;
    int aStep = block_dim;

    int bStart = block_dim * bx;
    int bStep = block_dim * n;

    int cSub = 0;
    
    __shared__ int As[1000][1000];
    __shared__ int Bs[1000][1000];

    for (int a = aStart, b = bStart; a <= aEnd; a += aStep, b += bStep){
        As[ty][tx] = A[a + n * ty + tx];
        Bs[ty][tx] = B[b + n * ty + tx];
        __syncthreads();

        for (int k = 0; k < block_dim; k++)
            cSub += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }

    C[c + n * ty + tx] = cSub;
}

__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n,
                       unsigned int block_dim){
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid(n/dimBlock.x, n/dimBlock.y);
    matmul<<<dimGrid, dimBlock>>>(A, B, C, n, block_dim);
    cudaDeviceSynchronize();
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n,
                       unsigned int block_dim){
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid(n/dimBlock.x, n/dimBlock.y);
    //matmul<float><<dimGrid, dimBlock>>(A, B, C, n, block_dim);
    cudaDeviceSynchronize();
}
__host__ void matmul_3(const double *A, const double *B, double *C,
                       unsigned int n, unsigned int block_dim){
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid(n/dimBlock.x, n/dimBlock.y);
    //matmul<double>l<<dimGrid, dimBlock>>(A, B, C, n, block_dim);
    cudaDeviceSynchronize();
}

#endif