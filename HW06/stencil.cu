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
// - The entire mask
// - The elements of image that are needed to compute the elements of output corresponding to the threads in the given block
// - The output image elements corresponding to the given block before it is written back to global memory
__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R){
    extern __shared__ float maskS[];    //why?
    __shared__ float outputBlock[blockDim.x];
    __shared__ float imageCurrent[R * 2 + 1];

    // index == true index | i == index within current block
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.x;

    //if (index <= n) return;   // leaving this out due to possible unsafe __syncthreads() calls later

    for (int j = -R; j < R; j++) {
        // load current image and mask(?):
        if (i + j < 0 || i + j >= n) imageCurrent[j] = 1;
        else imageCurrent[j] = image[i + j];
        maskS[j] = mask[j];
        __syncthreads();

        // update output:
        outputBlock[i] += imageCurrent[j] * maskS[j + R];
        __syncthreads();
    }

    // update actual output:
    output[index] = outputBlock[i];
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
    // block count is determined on the call of stencil
    stencil_kernel<<<1, threads_per_block, (R * 2 + 1) * sizeof(float)>>>(image, mask, output, n, R);
    cudaDeviceSynchronize();
}

#endif