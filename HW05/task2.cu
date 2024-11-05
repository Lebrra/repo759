#include <cuda.h>
#include <iostream>
using namespace std;

__global__ void algebraKernel(int* dA, int a) { 
    int index = threadIdx.x * (1 + blockIdx.x) + blockIdx.x;
    printf("index = %d | x = %d | y = %d \n", index, threadIdx.x, blockIdx.x);
    dA[index] = threadIdx.x * a + blockIdx.x;
}

int main() {
    int n = 16;
    int hA[n], *dA;
    //allocate memory on the device (GPU); zero out all entries in this device array
    cudaMalloc((void**)&dA, sizeof(int) * n);
    cudaMemset(dA, 0, n * sizeof(int));

    float r = static_cast <int> (rand() / static_cast <int> (RAND_MAX / 100)) - 50;
    printf("a = %d\n", r);

    algebraKernel<<<2, 8>>>(dA, r);
    cudaDeviceSynchronize();

    cudaMemcpy(&hA, dA, sizeof(int) * n, cudaMemcpyDeviceToHost);

    cout << "Results: " << endl;
    for (int i = 0; i < n; i++) cout << hA[i] << " ";
    cout << endl;

    cudaFree(dA);
    return 0;
}