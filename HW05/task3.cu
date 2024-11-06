#include <cuda.h>
#include <iostream>
#include <chrono>
#include "vscale.cuh"
using namespace std;

int main(int argc, char* argv[]) {
    int n = atoi(argv[1]);
    float hB[n], hA[n], *dB, *dA;

    auto start = chrono::steady_clock::now();

    // prepping threads and blocks:
    int t = 512;
    int b = (n + t - 1) / t;

    // randomization:
    srand(chrono::system_clock::now().time_since_epoch().count());

    // array initialization:

    for (int i = 0; i < n; i++){
        hA[i] = static_cast <float> (rand() / static_cast <float> (RAND_MAX / 20)) - 10;
        hB[i] = static_cast <float> (rand() / static_cast <float> (RAND_MAX / 2)) - 1;
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

    auto end = chrono::steady_clock::now();
	auto timePassed = chrono::duration_cast<std::chrono::microseconds>(end - start);

    cout << "Results: " << endl;
    cout << "element count:  \t" << n << endl;
    cout << "blocks:         \t" << b << endl;
	cout << "time to process:\t" << (timePassed.count() / 1000) << " milliseconds\n";
	cout << "first element:  \t" << hB[0] << endl;
	cout << "last element:   \t" << hB[n - 1] << endl << endl;
    cout << endl;

    cudaFree(dB);
    cudaFree(dA);
    return 0;
}