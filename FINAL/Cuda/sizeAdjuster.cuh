#ifndef SIZEADJUSTER_CUH
#define SIZEADJUSTER_CUH

using namespace std;

// adjusts 1D value to fix size bounds
__global__ void adjustValue(float* vertices, int vertexCount, float minX, float minY, float padding, float multiplier);

#endif