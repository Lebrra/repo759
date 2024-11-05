#include <cuda.h>
#include <iostream>
#include <random>
#include "vscale.cuh"
using namespace std;

__global__ void arrayInit(float *a, uniform_real_distribution<float> r, mt19937 g, int n){
    int index = threadIdx.x + blockIdx.x * 512;
    if (index < n) {
        a[index] = r(g);
    }
}

int main(int argc, char* argv[]) {
    int n = atoi(argv[1]);
    float hB[n], *dB, *dA;

    // prepping threads and blocks:
    int t = 512;
    int b = (n + t - 1) / t;
    printf("threads = %d | blocks = %d\n", t, b);

    // randomization:
    random_device entropy_source;
    mt19937 generator(entropy_source());
    uniform_real_distribution<float> distA(0., 20.);
    uniform_real_distribution<float> distB(0., 1.);

    // array initialization:
    cudaMalloc((void**)&dA, sizeof(float) * n);
    cudaMemset(dA, 0, n * sizeof(float));
    cudaMalloc((void**)&dB, sizeof(float) * n);
    cudaMemset(dB, 0, n * sizeof(float));
    arrayInit<<<b, t>>>(dA, distA, generator, n);
    arrayInit<<<b, t>>>(dB, distB, generator, n);
    cudaDeviceSynchronize();

    // do math:
    vscale<<<b, t>>>(dA, dB, n);
    cudaDeviceSynchronize();

    // results:
    cudaMemcpy(&hB, dB, sizeof(float) * n, cudaMemcpyDeviceToHost);

    cout << "Results: " << endl;
    for (int i = 0; i < 5; i++) {
        cout << " hB = " << hB[i] << endl;
    }
    cout << endl;

    cudaFree(dB);
    cudaFree(dA);
    return 0;
}