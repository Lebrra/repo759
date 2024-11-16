#include <cuda.h>
#include <iostream>
#include <chrono>
#include "matmul.cuh"
using namespace std;

int main(int argc, char* argv[]) {
    int n = atoi(argv[1]);
    int t = atoi(argv[2]);
    float hB[n], hA[n], hC[n*n], *dB, *dA, *dC;

    // randomization:
    srand(chrono::system_clock::now().time_since_epoch().count());

    // array initialization:
    for (int i = 0; i < n; i++){
        hA[i] = static_cast <float> (rand() / static_cast <float> (RAND_MAX / 2)) - 1;
        hB[i] = static_cast <float> (rand() / static_cast <float> (RAND_MAX / 2)) - 1;
    }

    auto start = chrono::steady_clock::now();

    cudaMalloc((void**)&dA, sizeof(float) * n);
    cudaMalloc((void**)&dB, sizeof(float) * n);
    cudaMalloc((void**)&dC, sizeof(float) * n * n);
    cudaMemcpy(dA, &hA, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, &hB, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, &hC, sizeof(float) * n * n, cudaMemcpyHostToDevice);

    // do math:
    matmul(dA, dB, dC, n, t);

    // results:
    cudaMemcpy(&hC, dC, sizeof(float) * n * n, cudaMemcpyDeviceToHost);

    auto end = chrono::steady_clock::now();
	auto timePassed = chrono::duration_cast<std::chrono::microseconds>(end - start);

    cout << "Results: " << endl;
    cout << "element count:  \t" << n << endl;
	cout << "time to process:\t" << (timePassed.count() / 1000) << " milliseconds\n";
	cout << "first element:  \t" << hC[0] << endl;
	cout << "last element:   \t" << hC[n*n - 1] << endl << endl;

    // print all:
    for (int i = 0; i < n * n; i++) {
		if ((i + 1) % n == 0) cout << "[";
        cout << hC[i];
        if ((i + 1) % n == 0) cout << "]" << endl;
        else cout << ", ";
	}

    cudaFree(dB);
    cudaFree(dA);
    cudaFree(dC);
    return 0;
}