#include <cuda.h>
#include <stdio.h>

__global__ void factorialKernel(int a) { 
    int b = 1;
    for (int i = a; i > 1; i++){
        b *= i;
    }
    printf("%d! = %d\n", a, b);
}

int main() {
    printf("printing...\n");
    for (int i = 1; i <= 8; i++){
        factorialKernel<<<1, 8>>>(i);
    }
    cudaDeviceSynchronize();
    return 0;
}