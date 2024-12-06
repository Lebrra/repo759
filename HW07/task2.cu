#include <cuda.h>
#include <iostream>
#include <chrono>
#include "reduce.cuh"
using namespace std;

int main(int argc, char* argv[]) {
    int n = atoi(argv[1]);
    int t = atoi(argv[2]);
    int blocks = (n + t - 1) / t / 2;
    
    float hInput[n], hOutput[blocks], *dInput, *dOutput;

    // randomization:
    srand(chrono::system_clock::now().time_since_epoch().count());

    // array initialization:
    for (int i = 0; i < n; i++){
        hInput[i] = static_cast <float> (rand() / static_cast <float> (RAND_MAX / 2)) - 1;
    }

    auto start = chrono::steady_clock::now();

    cudaMalloc((void**)&dInput, sizeof(float) * n);
    cudaMalloc((void**)&dOutput, sizeof(float) * blocks);
    cudaMemcpy(dInput, &hInput, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemset(dOutput, 0, sizeof(float) * blocks);

    // do math:
    reduce(&dInput, &dOutput, n, t);

    // results:
    cudaMemcpy(&hOutput, dOutput, sizeof(float) * blocks, cudaMemcpyDeviceToHost);

    auto end = chrono::steady_clock::now();
	auto timePassed = chrono::duration_cast<std::chrono::microseconds>(end - start);

    cout << "element count:  \t" << n << endl;
	cout << "time to process:\t" << (timePassed.count() / 1000) << " milliseconds\n";
	cout << "result:         \t" << hOutput[0] << endl;

    cudaFree(dInput);
    cudaFree(dOutput);

    return 0;
}