#include <cuda.h>
#include <iostream>
#include <random>
#include "vscale.cuh"
using namespace std;

__global__ void vscaleInt(const float *a, float *b, unsigned int n){
    int index = threadIdx.x + blockIdx.x * 512;
    printf("index = %d | n = %d\n", index, n);
    if (index < n) {
        printf("a = %f | b = %f | a*b = %f \n", a[index], b[index], a[index] * b[index]);
        b[index] *= a[index];
    }
}

int main(int argc, char* argv[]) {
    int n = atoi(argv[1]);

    float hB[n], *dB, *dA;

    random_device entropy_source;
    mt19937 generator(entropy_source());
    uniform_real_distribution<float> distA(0., 20.);
    uniform_real_distribution<float> distB(0., 1.);

    cudaMalloc((void**)&dA, sizeof(float) * n);
    cudaMemset(dA, distA(generator), n * sizeof(float));
    cudaMalloc((void**)&dB, sizeof(float) * n);
    cudaMemset(dB, distB(generator), n * sizeof(float));

    int threads = 512;
    int blocks = (n + threads - 1) / threads;
    printf("threads = %d | blocks = %d\n", threads, blocks);

    vscaleInt<<<blocks, threads>>>(dA, dB, n);
    cudaDeviceSynchronize();

    cudaMemcpy(&hB, dB, sizeof(float) * n, cudaMemcpyDeviceToHost);

    cout << "Results: " << endl;
    for (int i = 0; i < 5; i++) {
        cout << " dB = " << hB[i] << endl;
    }
    cout << endl;

    cudaFree(dB);
    return 0;
}