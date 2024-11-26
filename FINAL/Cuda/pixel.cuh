#ifndef PIXEL_CUH
#define PIXEL_CUH

using namespace std;

// barycentric calculation between 3 2D points
__device__ float barycentric(float p1X, float p1Y, float p2X, float p2Y, float p3X, float p3Y);

// iterates through every triangle to see if this pixel is within any given triangle
__global__ void inTriangle(float* triangle, int* results, int triangleValue);

#endif