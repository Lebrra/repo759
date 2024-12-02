#ifndef MATMUL_CUH

// (the difference is types of data)

__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n,
                       unsigned int block_dim){
    
    // todo: convert this to using the grid-based indexing in the powerpoints, then map to other 2 implementation types
    int iIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (iIndex >= n*n) return;

    for (int k = 0; k < n; k++){
        int jIndex = (iIndex / n) * n + k;
        int kIndex = k * n + (iIndex % n);
        C[iIndex] += A[jIndex] * B[kIndex];
    }
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n,
                       unsigned int block_dim){

}
__host__ void matmul_3(const double *A, const double *B, double *C,
                       unsigned int n, unsigned int block_dim){
    
}

#endif