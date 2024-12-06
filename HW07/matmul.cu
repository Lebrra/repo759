#include <cuda.h>
#include <iostream> 
using namespace std; 

#ifndef MATMUL_CUH

// (the difference is types of data)

template <typename T>
__global__ void matmul(const T *A, const T *B, T *C, unsigned int n, unsigned int block_dim){
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int c = n * (block_dim * by + ty) + (block_dim * bx + tx);
    if (c >= n*n) return;

    int aStart = n * block_dim * by;
    int aEnd = aStart + n - 1;
    int aStep = block_dim;

    int bStart = block_dim * bx;
    int bStep = block_dim * n;

    T cSub = 0;
    
    extern __shared__ char shared[];
    T* As = (T*)shared;
    T* Bs = (T*)&As[block_dim*block_dim];  

    for (int a = aStart, b = bStart; a <= aEnd; a += aStep, b += bStep){
        As[ty * block_dim + tx] = A[a + n * ty + tx];
        Bs[ty * block_dim + tx] = B[b + n * ty + tx];
        __syncthreads();

        for (int k = 0; k < block_dim; k++){
            cSub += As[ty * block_dim + k] * Bs[k * block_dim + tx];
        }
            
        __syncthreads();
    }

    C[c] = cSub;
}

__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n,
                       unsigned int block_dim){
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    matmul<int><<<dimGrid, dimBlock, block_dim*block_dim*2 * sizeof(int)>>>(A, B, C, n, block_dim);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n,
                       unsigned int block_dim){
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    matmul<float><<<dimGrid, dimBlock, block_dim*block_dim*2 * sizeof(float)>>>(A, B, C, n, block_dim);
    cudaDeviceSynchronize();
}
__host__ void matmul_3(const double *A, const double *B, double *C,
                       unsigned int n, unsigned int block_dim){
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    matmul<double><<<dimGrid, dimBlock, block_dim*block_dim*2 * sizeof(double)>>>(A, B, C, n, block_dim);
    cudaDeviceSynchronize();
}

#endif