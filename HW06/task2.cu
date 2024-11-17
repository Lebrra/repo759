#include <cuda.h>
#include <iostream>
#include <chrono>
#include "stencil.cuh"
using namespace std;

int main(int argc, char* argv[]) {
    int n = atoi(argv[1]);
    int R = atoi(argv[2]);
    int t = atoi(argv[3]);
    const int RExpanded = 2 * R + 1;
    float hImage[n], hMask[RExpanded], hOutput[n], *dImage, *dMask, *dOutput;

    // randomization:
    srand(chrono::system_clock::now().time_since_epoch().count());

    // array initialization:
    for (int i = 0; i < n; i++){
        hImage[i] = static_cast <float> (rand() / static_cast <float> (RAND_MAX / 2)) - 1;
    }
    for (int i = 0; i < RExpanded; i++){
        hMask[i] = static_cast <float> (rand() / static_cast <float> (RAND_MAX / 2)) - 1;
    }

    auto start = chrono::steady_clock::now();

    cudaMalloc((void**)&dImage, sizeof(float) * n);
    cudaMalloc((void**)&dMask, sizeof(float) * RExpanded);
    cudaMalloc((void**)&dOutput, sizeof(float) * n);
    cudaMemcpy(dImage, &hImage, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dMask, &hMask, sizeof(float) * RExpanded, cudaMemcpyHostToDevice);
    cudaMemset(dOutput, 0, n * sizeof(float));

    // call stencil
    stencil(dImage, dMask, dOutput, n, R, t);

    // results:
    cudaMemcpy(&hOutput, dOutput, sizeof(float) * n, cudaMemcpyDeviceToHost);

    auto end = chrono::steady_clock::now();
	auto timePassed = chrono::duration_cast<std::chrono::microseconds>(end - start);

    cout << "Results: " << endl;
    cout << "element count:  \t" << n << endl;
    cout << "R value:        \t" << R << endl;
	cout << "time to process:\t" << (timePassed.count() / 1000) << " milliseconds\n";
	cout << "first element:  \t" << hOutput[0] << endl;
	cout << "last element:   \t" << hOutput[n - 1] << endl << endl;

    cudaFree(dImage);
    cudaFree(dMask);
    cudaFree(dOutput);
    return 0;
}