#include <cuda.h>
#include <iostream>
#include <chrono>
#include "vscale.cuh"
using namespace std;

int main(int argc, char* argv[]) {
    int n = atoi(argv[1]);
    float hB[n], hA[n], *dB, *dA;

    // prepping threads and blocks:
    int t = 512;
    int b = (n + t - 1) / t;
    printf("threads = %d | blocks = %d\n", t, b);

    // randomization:
    srand(chrono::system_clock::now().time_since_epoch().count());

    // array initialization:

    for (int i = 0; i < n; i++){
        hA[i] = static_cast <float> (rand() / static_cast <float> (RAND_MAX / 20)) - 10;
        hB[i] = static_cast <float> (rand() / static_cast <float> (RAND_MAX / 2)) - 1;

        if (i < 5){
            cout << " dA = " << hA[i] << " | dB = " << hB[i] << endl;
        }
    }

    cudaMalloc((void**)&dA, sizeof(float) * n);
    cudaMalloc((void**)&dB, sizeof(float) * n);
    cudaMemcpy(dA, &hA, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, &hB, sizeof(float) * n, cudaMemcpyHostToDevice);

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