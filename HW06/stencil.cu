#include <cuda.h>
#include <stdio.h>

#ifndef STENCIL_CUH

// Computes the convolution of image and mask, storing the result in output.
// Each thread should compute _one_ element of the output matrix.
// Shared memory should be allocated _dynamically_ only.
//
// image is an array of length n.
// mask is an array of length (2 * R + 1).
// output is an array of length n.
// All of them are in device memory
//
// Assumptions:
// - 1D configuration
// - blockDim.x >= 2 * R + 1
//
// The following should be stored/computed in shared memory:
// - The entire mask    ?????
// - The elements of image that are needed to compute the elements of output corresponding to the threads in the given block
// - The output image elements corresponding to the given block before it is written back to global memory
__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R){
    extern __shared__ float allSharedData[];

    // index == true index | i == index within current block
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.x;
    int RExpand = R * 2 + 1;

    if (index >= n) return;

    // pointers to shared arrays:
    float* maskPointer = (float*)allSharedData;                 // length = RExpand
    float* imagePointer = (float*)&maskPointer[RExpand];        // length = blockDim.x
    float* outputPointer = (float*)&imagePointer[blockDim.x];   // length = blockDim.x
    
    // set shared values:
    imagePointer[i] = image[index];
    if (i < RExpand) maskPointer[i] = mask[i];
    outputPointer[i] = 0;
    __syncthreads();

    // compute:
    for (int j = 0; j < R*2; j++){
        if (i + j < 0 || i + j >= n) outputPointer[i] += maskPointer[j];
        else outputPointer[i] += imagePointer[i + j] * maskPointer[j];
    }
    __syncthreads();

    output[index] = outputPointer[i];
}

// Makes one call to stencil_kernel with threads_per_block threads per block.
// You can consider following the kernel call with cudaDeviceSynchronize (but if you use
// cudaEventSynchronize to time it, that call serves the same purpose as cudaDeviceSynchronize).
//
// Assumptions:
// - threads_per_block >= 2 * R + 1
__host__ void stencil(const float* image,
                      const float* mask,
                      float* output,
                      unsigned int n,
                      unsigned int R,
                      unsigned int threads_per_block){
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    int sharedCount = (R * 2 + 1) + threads_per_block * 2;
    stencil_kernel<<<blocks, threads_per_block, sharedCount * sizeof(float)>>>(image, mask, output, n, R);
    cudaDeviceSynchronize();
}

#endif