#include <cuda.h>
#include <iostream>
#include <chrono>
#include "matmul.cuh"
using namespace std;

template <typename T>
void doMatmul(int n, int blockSize){
    T hB[n*n], hA[n*n], hC[n*n], *dB, *dA, *dC;

    // randomization:
    srand(chrono::system_clock::now().time_since_epoch().count());

    // array initialization:
    for (int i = 0; i < n*n; i++){
        hA[i] = static_cast <T> (rand() / static_cast <T> (RAND_MAX / 20)) - 10;
        hB[i] = static_cast <T> (rand() / static_cast <T> (RAND_MAX / 20)) - 10;
    }

    auto start = chrono::steady_clock::now();

    cudaMalloc((void**)&dA, sizeof(T) * n * n);
    cudaMalloc((void**)&dB, sizeof(T) * n * n);
    cudaMalloc((void**)&dC, sizeof(T) * n * n);
    cudaMemcpy(dA, &hA, sizeof(T) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, &hB, sizeof(T) * n * n, cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, n * n * sizeof(T));

    // do math:
    cout << "Switch time\n";
    switch(sizeof(T)){
        case sizeof(int):
            cout << "Calculating type: int" << endl;
            matmul_1(dA, dB, dC, n, blockSize);
            break;
        case sizeof(float):
            cout << "Calculating type: float" << endl;
            matmul_2(dA, dB, dC, n, blockSize);
            break;
        case sizeof(double):
            cout << "Calculating type: double" << endl;
            matmul_3(dA, dB, dC, n, blockSize);
            break;
        default:
            cout << "Invalid type to process matmul.\n";
            return;
    }

    // results:
    cudaMemcpy(&hC, dC, sizeof(T) * n * n, cudaMemcpyDeviceToHost);

    auto end = chrono::steady_clock::now();
	auto timePassed = chrono::duration_cast<std::chrono::microseconds>(end - start);

    //cout << "Results of type " << typeof(T) << ":" << endl;
    cout << "element count:  \t" << n << endl;
	cout << "time to process:\t" << (timePassed.count() / 1000) << " milliseconds\n";
	cout << "first element:  \t" << hC[0] << endl;
	cout << "last element:   \t" << hC[n*n - 1] << endl << endl;

    cudaFree(dB);
    cudaFree(dA);
    cudaFree(dC);
}

int main(int argc, char* argv[]) {
    int n = atoi(argv[1]);
    int block = atoi(argv[2]);
    int type = atoi(argv[3]);
    
    cout << "I should execute matmul_" << type << " with an " << n << "x" << n << " matrix and " << block << " blocks\n";

    return 0;
}