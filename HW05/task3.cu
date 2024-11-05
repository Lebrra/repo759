#include <cuda.h>
#include <iostream>
#include <random>
#include "vscale.cuh"
using namespace std;

int main(int argc, char* argv[]) {
    int n = atoi(argv[1]);

    float hB[n], *dB, *dA;

    random_device entropy_source;
    mt19937 generator(entropy_source());
    uniform_float_distribution<float> distA(0., 20.);
    uniform_float_distribution<float> distB(0., 1.);

    cudaMalloc((void**)&dA, sizeof(float) * n);
    cudaMemset(dA, distA(generator), n * sizeof(float));
    cudaMalloc((void**)&dB, sizeof(float) * n);
    cudaMemset(dB, distB(generator), n * sizeof(float));

    for (int i = 0; i < 5; i++) {
        cout << dA[i] << " * " << dB[i] << endl;
    }

    int blocks = (n + 512 - 1) / 512;
    algebraKernel<<<blocks, 512>>>(dA, dB, n);
    cudaDeviceSynchronize();

    cudaMemcpy(&hB, dB, sizeof(float) * n, cudaMemcpyDeviceToHost);

    cout << "Results: " << endl;
    for (int i = 0; i < 5; i++) {
        cout << " dB = " << dB[i] << endl;
    }
    cout << endl;

    cudaFree(dB);
    return 0;
}