
#include <cuda.h>
#include <stdio.h>
using namespace std;

#ifndef PIXEL_CUH

// first triangle: 
// - point 1 = (points[triangle[0]], points[triangle[0] + 1], points[triangle[0] + 2])
// - point 2 = (points[triangle[0 + 1]], points[triangle[0 + 1] + 1], points[triangle[0 + 1] + 2])
// - point 3 = (points[triangle[0 + 2]], points[triangle[0 + 2] + 1], points[triangle[0 + 2] + 2])

//https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
__device__ float barycentric(float p1X, float p1Y, float p2X, float p2Y, float p3X, float p3Y){
    return (p1X - p3X) * (p2Y - p3Y) - (p2X - p3X) * (p1Y - p3Y);
}

// block will always be one line of pixels (assuming size < 1024)
__global__ void inTriangle(float* triangle, int* results, int triangleValue){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (results[index] > 0) return;

    int x = threadIdx.x;
    int y = blockIdx.x;
    if (index < 10 || index == 256*256-5) printf("Analyzing pixel (%d, %d) - index %d\n", x, y, index);

    float b1 = barycentric(x, y, triangle[0], triangle[1], triangle[2], triangle[3]);
    float b2 = barycentric(x, y, triangle[2], triangle[3], triangle[4], triangle[5]);
    float b3 = barycentric(x, y, triangle[4], triangle[5], triangle[0], triangle[1]);

    bool neg = b1 <= 0 && b2 <= 0 && b3 <= 0;
    bool pos = b1 >= 0 && b2 >= 0 && b3 >= 0;

    if (neg || pos) results[index] = triangleValue;
}

#endif