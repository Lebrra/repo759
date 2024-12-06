#ifndef REDUCE_CUH

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n){
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // guarding for if n is odd
    if (i + blockDim.x < n) shared[tid] = g_idata[i] + g_idata[i + blockDim.x];
    else shared[tid] = g_idata[i];
    __syncthreads();


    for (int s = blockDim.x/2; s > 0; s >>= 2) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = shared[0];
}

__host__ void reduce(float **input, float **output, unsigned int N,
                     unsigned int threads_per_block){
    int blocks = (N + threads_per_block - 1) / threads_per_block / 2;
    reduce_kernel<<<blocks, threads_per_block, blocks>>>(*input, *output, N);
    cudaDeviceSynchronize();
}

#endif