#include <cuda.h>
#include <iostream>
#include <chrono>
#include "matmul.cuh"
using namespace std;

void doInt(int n, int blockSize){
    int hB[n*n], hA[n*n], hC[n*n], *dB, *dA, *dC;

    // randomization:
    srand(chrono::system_clock::now().time_since_epoch().count());

    // array initialization:
    for (int i = 0; i < n*n; i++){
        hA[i] = static_cast <int> (rand() / static_cast <int> (RAND_MAX / 20)) - 10;
        hB[i] = static_cast <int> (rand() / static_cast <int> (RAND_MAX / 20)) - 10;
    }

    auto start = chrono::steady_clock::now();

    cudaMalloc((void**)&dA, sizeof(int) * n * n);
    cudaMalloc((void**)&dB, sizeof(int) * n * n);
    cudaMalloc((void**)&dC, sizeof(int) * n * n);
    cudaMemcpy(dA, &hA, sizeof(int) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, &hB, sizeof(int) * n * n, cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, n * n * sizeof(int));

    // do math:
    cout << "Calculating type: int" << endl;
    matmul_1(dA, dB, dC, n, blockSize);

    // results:
    cudaMemcpy(&hC, dC, sizeof(int) * n * n, cudaMemcpyDeviceToHost);

    auto end = chrono::steady_clock::now();
	auto timePassed = chrono::duration_cast<std::chrono::microseconds>(end - start);

    cout << "element count:  \t" << n << endl;
	cout << "time to process:\t" << (timePassed.count() / 1000) << " milliseconds\n";
	cout << "first element:  \t" << hC[0] << " (" << hA[0] << ", " << hB[0] << ")" << endl;
	cout << "last element:   \t" << hC[n*n - 1] << " (" << hA[n*n - 1] << ", " << hB[n*n - 1] << ")" << endl << endl;
    cudaFree(dB);
    cudaFree(dA);
    cudaFree(dC);
}

void doFloat(int n, int blockSize){
    float hB[n*n], hA[n*n], hC[n*n], *dB, *dA, *dC;

    // randomization:
    srand(chrono::system_clock::now().time_since_epoch().count());

    // array initialization:
    for (int i = 0; i < n*n; i++){
        hA[i] = static_cast <float> (rand() / static_cast <float> (RAND_MAX / 20)) - 10;
        hB[i] = static_cast <float> (rand() / static_cast <float> (RAND_MAX / 20)) - 10;
    }

    auto start = chrono::steady_clock::now();

    cudaMalloc((void**)&dA, sizeof(float) * n * n);
    cudaMalloc((void**)&dB, sizeof(float) * n * n);
    cudaMalloc((void**)&dC, sizeof(float) * n * n);
    cudaMemcpy(dA, &hA, sizeof(float) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, &hB, sizeof(float) * n * n, cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, n * n * sizeof(float));

    // do math:
    cout << "Calculating type: float" << endl;
    matmul_2(dA, dB, dC, n, blockSize);

    // results:
    cudaMemcpy(&hC, dC, sizeof(float) * n * n, cudaMemcpyDeviceToHost);

    auto end = chrono::steady_clock::now();
	auto timePassed = chrono::duration_cast<std::chrono::microseconds>(end - start);

    cout << "element count:  \t" << n << endl;
	cout << "time to process:\t" << (timePassed.count() / 1000) << " milliseconds\n";
	cout << "first element:  \t" << hC[0] << " (" << hA[0] << ", " << hB[0] << ")" << endl;
	cout << "last element:   \t" << hC[n*n - 1] << " (" << hA[n*n - 1] << ", " << hB[n*n - 1] << ")" << endl << endl;

    cudaFree(dB);
    cudaFree(dA);
    cudaFree(dC);
}

void doDouble(int n, int blockSize){
    double hB[n*n], hA[n*n], hC[n*n], *dB, *dA, *dC;

    // randomization:
    srand(chrono::system_clock::now().time_since_epoch().count());

    // array initialization:
    for (int i = 0; i < n*n; i++){
        hA[i] = static_cast <double> (rand() / static_cast <double> (RAND_MAX / 20)) - 10;
        hB[i] = static_cast <double> (rand() / static_cast <double> (RAND_MAX / 20)) - 10;
    }

    auto start = chrono::steady_clock::now();

    cudaMalloc((void**)&dA, sizeof(double) * n * n);
    cudaMalloc((void**)&dB, sizeof(double) * n * n);
    cudaMalloc((void**)&dC, sizeof(double) * n * n);
    cudaMemcpy(dA, &hA, sizeof(double) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, &hB, sizeof(double) * n * n, cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, n * n * sizeof(double));

    // do math:
    cout << "Calculating type: double" << endl;
    matmul_3(dA, dB, dC, n, blockSize);

    // results:
    cudaMemcpy(&hC, dC, sizeof(double) * n * n, cudaMemcpyDeviceToHost);

    auto end = chrono::steady_clock::now();
	auto timePassed = chrono::duration_cast<std::chrono::microseconds>(end - start);

    cout << "element count:  \t" << n << endl;
	cout << "time to process:\t" << (timePassed.count() / 1000) << " milliseconds\n";
	cout << "first element:  \t" << hC[0] << " (" << hA[0] << ", " << hB[0] << ")" << endl;
	cout << "last element:   \t" << hC[n*n - 1] << " (" << hA[n*n - 1] << ", " << hB[n*n - 1] << ")" << endl << endl;

    cudaFree(dB);
    cudaFree(dA);
    cudaFree(dC);
}

int main(int argc, char* argv[]) {
    int n = atoi(argv[1]);
    int block = atoi(argv[2]);
    int type = atoi(argv[3]);
    
    cout << "I should execute matmul_" << type << " with an " << n << "x" << n << " matrix and " << block << " blocks\n";
    switch(type){
        case 1: 
            doInt(n, block);
            break;
        case 2:
            doFloat(n, block);
            break;
        case 3:
            doDouble(n, block);
            break;
    }

    return 0;
}