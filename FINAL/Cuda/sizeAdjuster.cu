#ifndef SIZEADJUSTER_CUH

#include <cuda.h>
using namespace std;

__global__ void adjustValue(float* vertices, int vertexCount, float minX, float minY, float padding, float multiplier){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= vertexCount || index % 3 == 2) return;
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

__host__ void adjustSize(float* vertices, int vertexCount, float size, float padding){
    float minX = 0;
    float maxX = 0;
    float minY = 0;
    float maxY = 0;

    // calculate min and max -es
    for (int i = 0; i < vertexCount; i++) {
        int x = i * 3;  // x + 1 = y

        if (vertices[x] < minX) minX = vertices[x];
        if (vertices[x] > maxX) maxX = vertices[x];
        if (vertices[x + 1] < minY) minY = vertices[x + 1];
        if (vertices[x + 1] > maxY) maxY = vertices[x + 1];
    }

    // create multiplier based off larger difference
    float pointsWidth = maxX - minX;
    float pointsHeight = maxY - minY;

    float multiplier;
    if (pointsWidth > pointsHeight) {
        multiplier = (size - padding*2) / pointsWidth;
    }
    else { 
        multiplier = (size - padding*2) / pointsHeight;
    }

    // apply multiplier to all points (and offset if any points are negative)
    int blocks = ((vertexCount*3) + 256 - 1) / 256;
    print("applying adjustments using block count: %d\n", blocks);
    adjustValue<<<blocks, 256>>>(vertices, vertexCount, minX, minY, padding, multiplier);
}

#endif