#include <cuda.h>
#include <stdio.h>

#ifndef VSCALE_CUH
__global__ void vscale(const float *a, float *b, unsigned int n){
    int index = threadIdx.x + blockIdx.x * 16;
    if (index < n) {
        b[index] *= a[index];
    }
}

#endif