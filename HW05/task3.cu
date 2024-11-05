#include <cuda.h>
#include <iostream>
#include <chrono>
#include "vscale.cuh"
using namespace std;

int main(int argc, char* argv[]) {
    int n = atoi(argv[1]);
    float hB[n], *dB, *dA;

    // prepping threads and blocks:
    int t = 512;
    int b = (n + t - 1) / t;
    printf("threads = %d | blocks = %d\n", t, b);

    // randomization:
    srand(chrono::system_clock::now().time_since_epoch().count());

    // array initialization:
    dA = (float*)malloc(sizeof(float)*n);
    dB = (float*)malloc(sizeof(float)*n);

    for (int i = 0; i < n; i++){
        dA[i] = static_cast <float> (rand() / static_cast <float> (RAND_MAX / 20)) - 10;
        dB[i] = static_cast <float> (rand() / static_cast <float> (RAND_MAX / 2)) - 1;
    }

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