#include <cuda.h>
#include <iostream>
#include <random>
using namespace std;

__global__ void algebraKernel(int* dA, int a) { 
    printf("a = %d", a);
    int index = threadIdx.x + blockIdx.x * 8;
    dA[index] = threadIdx.x * a + blockIdx.x;
}

int main() {
    int n = 16;
    int hA[n], *dA;
    //allocate memory on the device (GPU); zero out all entries in this device array
    cudaMalloc((void**)&dA, sizeof(int) * n);
    cudaMemset(dA, 0, n * sizeof(int));

    mt19937 generator(12354);
    uniform_int_distribution<int> dist(0., 100.);
    auto r = dist(generator);

    algebraKernel<<<2, 8>>>(dA, r);
    cudaDeviceSynchronize();

    cudaMemcpy(&hA, dA, sizeof(int) * n, cudaMemcpyDeviceToHost);

    cout << "Results: " << endl;
    for (int i = 0; i < n; i++) cout << hA[i] << " ";
    cout << endl;

    cudaFree(dA);
    return 0;
}