#include <cuda.h>
#include <iostream>
#include <random>
#include "vscale.cuh"
using namespace std;

__global__ void arrayInit(float *a, float r, int n){
    int index = threadIdx.x + blockIdx.x * 512;
    if (index < n) {
        a[index] = r;
    }
}

int main(int argc, char* argv[]) {
    int n = 16;
    float hA[n], *dA;

    cudaMalloc((void**)&dA, sizeof(float) * n);
    cudaMemset(dA, 0, n * sizeof(float));
    arrayInit<<<2, 8>>>(dA, 5., n);

    //random_device entropy_source;
    //mt19937 generator(entropy_source());
    //uniform_real_distribution<float> dist(0, 100);
    //auto r = dist(generator);

    vscale<<<2, 8>>>(dA, dA, n);
    cudaDeviceSynchronize();

    cudaMemcpy(&hA, dA, sizeof(float) * n, cudaMemcpyDeviceToHost);

    cout << "Results: " << endl;
    for (int i = 0; i < n; i++) cout << hA[i] << " ";
    cout << endl;

    cudaFree(dA);
    return 0;
}