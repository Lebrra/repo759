#include <cuda.h>
#include <stdio.h>

__global__ void factorialKernel() { 
    int a = threadIdx.x + 1;
    int b = 1;
    for (int i = 2; i <= a; i++) {
        b *= i;
    }
    printf("%d! = %d\n", a, b);
}

int main() {
    factorialKernel<<<1, 8>>>();
    cudaDeviceSynchronize();
    return 0;
}