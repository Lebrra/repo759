#include <cuda.h>
#include <stdio.h>

#ifndef VSCALE_CUH
__global__ void vscale(const float *a, float *b, unsigned int n){
    int index = threadIdx.x + blockIdx.x * 512;
    printf("index = %d | n = %d\n", index, n);
    if (index < n) {
        if (index < 5) printf("a = %f | b = %f | a*b = %f \n", a[index], b[index], a[index] * b[index]);
        b[index] *= a[index];
    }
}

#endif