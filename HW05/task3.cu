#include <cuda.h>
#include <iostream>
#include <random>
#include "vscale.cuh"
using namespace std;

int main(int argc, char* argv[]) {
    int n = 16;
    float hA[n], *dA;

    cudaMalloc((void**)&dA, sizeof(float) * n);
    cudaMemset(dA, 5., n * sizeof(float));

    random_device entropy_source;
    mt19937 generator(entropy_source());
    uniform_real_distribution<float> dist(0, 100);
    auto r = dist(generator);

    vscale<<<2, 8>>>(dA, dA, n);
    cudaDeviceSynchronize();

    cudaMemcpy(&hA, dA, sizeof(float) * n, cudaMemcpyDeviceToHost);

    cout << "Results: " << endl;
    for (int i = 0; i < n; i++) cout << hA[i] << " ";
    cout << endl;

    cudaFree(dA);
    return 0;
}