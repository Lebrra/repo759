#ifndef REDUCE_CUH

const int MAX_BLOCKS = 65535;

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n){
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // guarding for if n is odd
    shared[tid] = g_idata[i + n] + g_idata[i + blockDim.x + n];
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

    if (blocks > MAX_BLOCKS){
        int iterations = (blocks + MAX_BLOCKS - 1) / MAX_BLOCKS;

        float *dPartialOutput;
        cudaMalloc((void**)&dPartialOutput, sizeof(float) * MAX_BLOCKS);
        cudaMemset(dPartialOutput, 0, sizeof(float) * MAX_BLOCKS);
        
        for (int i = 0; i < iterations - 1; i++){
            // do max blocks 
            reduce_kernel<<<MAX_BLOCKS, threads_per_block, MAX_BLOCKS>>>(*input, dPartialOutput, i * MAX_BLOCKS);
            cudaDeviceSynchronize();

            // copy back section to full output:
            cudaMemcpy((float*)&output[i * MAX_BLOCKS], dPartialOutput, sizeof(float), cudaMemcpyDeviceToHost);
        }

        // do final with remaining blocks
        int remainingBlocks = blocks % MAX_BLOCKS;
        if (remainingBlocks == 0) remainingBlocks = MAX_BLOCKS;

        reduce_kernel<<<remainingBlocks, threads_per_block, remainingBlocks>>>(*input, dPartialOutput, (iterations - 1) * MAX_BLOCKS);
        cudaDeviceSynchronize();

        // copy back section to full output:
        cudaMemcpy((float*)&output[(iterations - 1) * MAX_BLOCKS], dPartialOutput, sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(dPartialOutput);

    }
    else {
        reduce_kernel<<<blocks, threads_per_block, blocks>>>(*input, *output, 0);
        cudaDeviceSynchronize();
    }
}

#endif