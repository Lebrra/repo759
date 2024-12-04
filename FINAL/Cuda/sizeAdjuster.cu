#ifndef SIZEADJUSTER_CUH

#include <cuda.h>
#include <stdio.h>
using namespace std;

__global__ void adjustValue(float* vertices, int vertexCount, float minX, float minY, float padding, float multiplier){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= vertexCount*3 || index % 3 == 2) return;
    // ignore z for now its not being used

    if (index % 3 == 0 && minX < 0){
        vertices[index] -= minX;
    }
    else if (index % 3 == 1 && minY < 0){
        vertices[index] -= minY; 
    }

    vertices[index] *= multiplier;
    vertices[index] += padding;
}

#endif