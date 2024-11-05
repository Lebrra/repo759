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
    uniform_real_distribution<float> distA(0., 20.);
    uniform_real_distribution<float> distB(0., 1.);

    cudaMalloc((void**)&dA, sizeof(float) * n);
    cudaMemset(dA, 0, n * sizeof(float));
    cudaMalloc((void**)&dB, sizeof(float) * n);
    cudaMemset(dB, 0, n * sizeof(float));
    
    // set dA and dB to random values:
    for(int i = 0; i < n; i++){
        dA[i] = distA(generator);
        dB[i] = distB(generator);
    }

    int t = 512;
    int b = (n + t - 1) / t;
    printf("threads = %d | blocks = %d\n", t, b);

    vscale<<<b, t>>>(dA, dB, n);
    cudaDeviceSynchronize();

    cudaMemcpy(&hB, dB, sizeof(float) * n, cudaMemcpyDeviceToHost);

    cout << "Results: " << endl;
    for (int i = 0; i < 5; i++) {
        cout << " hB = " << hB[i] << endl;
    }
    cout << endl;

    cudaFree(dB);
    return 0;
}