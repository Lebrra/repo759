#include <cuda.h>
#include <stdio.h>
using namespace std;

__global__ void factorialKernel(int a) { 
    int b = 1;
    for (int i = a; i > 1; i++){
        b *= i;
    }
    printf(b);
    //printf("%d! = %d", a, b);
}

int main() {
    printf("printing...\n")
    for (int i = 1; i <= 8; i++){
        factorialKernel<<<1, 8>>>(i);
    }
    cudaDeviceSynchronize();
    return 0;
}