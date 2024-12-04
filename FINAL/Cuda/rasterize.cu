// inspiration: https://github.com/ssloy/tinyrenderer/wiki/Lesson-2:-Triangle-rasterization-and-back-face-culling

#include <cuda.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include "pixel.cuh"
//#include "sizeAdjuster.cuh"
#include "filehandler.cuh"

using namespace std;

// hardset output height and width
const int definedSize = 256;
const float padding = 10;

__global__ void adjustValue(float* vertices, int vertexCount, float minX, float minY, float padding, float multiplier){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Printing in kernel %d", index);
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
    printf("within adjustSize...\n");
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
    printf("completed min/max...\n");

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

    printf("ready to calculate blocks...\n");

    // apply multiplier to all points (and offset if any points are negative)
    int blocks = ((vertexCount*3) + 256 - 1) / 256;
    printf("applying adjustments using block count: %d\n", blocks);
    adjustValue<<<blocks, 256>>>(vertices, vertexCount, minX, minY, padding, multiplier);
    cudaDeviceSynchronize();
}

int main(int argc, char** argv) {
    if (argc <= 1){
        cout << "Please include a filename (excluding the '_vertices.txt' or '_faces.txt') to rasterize!";
        return 0;
    }
    // else I could add optional size values, but let's just hardset it to 256...
    if (definedSize > 1024 || definedSize <= 0){
        cout << "Invalid image size; please pick a value between 1 and 1024";
        return 0;
    }

    auto start = chrono::steady_clock::now();

    // seeding with chrono so I don't have to include the ctime library:
	srand(chrono::system_clock::now().time_since_epoch().count());

    string fileName = argv[1];
    cout << "Processing file: " << fileName << endl;

    // gather data:
    int vertCount = getVertexCount(fileName);
    int triangleCount = getFaceCount(fileName);

    float* vertices = (float*)malloc(sizeof(float) * vertCount * 3);
    int* faces = (int*)malloc(sizeof(int) * triangleCount * 3);
    //float vertices[vertCount*3]; 
    //int faces[triangleCount*3];
    //for(int i = 0; i < max(vertCount, triangleCount)*3; i++){
    //    if (i < vertCount*3) vertices[i] = i;
    //    if (i < triangleCount*3) faces[i] = i;
    //}
    float *dVerts;
    
    readVertices(fileName, vertices);
    readFaces(fileName, faces);

    cout << "This file has " << vertCount << " vertices and " << triangleCount << " triangles!" << endl;

    cout << "Adjusting to size..." << endl;
    cudaMalloc((void**)&dVerts, sizeof(float) * vertCount*3);
    cudaMemcpy(dVerts, &vertices, sizeof(float) * vertCount*3, cudaMemcpyHostToDevice);

    cout << "Calling adjust..." << endl;
    adjustSize(dVerts, vertCount, definedSize, padding);
    cout << "Completed adjusting..." << endl;
    cudaMemcpy(&vertices, dVerts, sizeof(float) * vertCount*3, cudaMemcpyDeviceToHost);
    cudaFree(dVerts);

    int pointTests[definedSize * definedSize], *dPoints;
    cudaMalloc((void**)&dPoints, sizeof(int) * definedSize * definedSize);
    cudaMemset(dPoints, 0, sizeof(int) * definedSize * definedSize);

    // do math
    int validTriangles = 0;
    cout << "Comparing pixels with triangles..." << endl;
    float triangle[6], *dTri;
    cudaMalloc((void**)&dTri, sizeof(float) * 6);
    for(int tri = 0; tri < triangleCount; tri++){
        int face1 = faces[tri * 3];
        int face2 = faces[tri * 3 + 1];
        int face3 = faces[tri * 3 + 2];

        triangle[0] = vertices[face1*3];
        triangle[1] = vertices[face1*3 + 1];
        triangle[2] = vertices[face2*3];
        triangle[3] = vertices[face2*3 + 1];
        triangle[4] = vertices[face3*3];
        triangle[5] = vertices[face3*3 + 1];

        // validate triangle: (if two points are only different in the z direction then let's just skip it)
        if ((triangle[0] == triangle[2] && triangle[1] == triangle[3]) ||
            (triangle[0] == triangle[4] && triangle[1] == triangle[5]) ||
            (triangle[2] == triangle[4] && triangle[3] == triangle[5])){
                continue;
            }
        validTriangles++;
        cudaMemcpy(dTri, &triangle, sizeof(float) * 6, cudaMemcpyHostToDevice);

        // do parallelism here
        inTriangle<<<definedSize, definedSize>>>(dTri, dPoints, validTriangles);
    }
    cudaFree(dTri);

    cudaMemcpy(&pointTests, dPoints, sizeof(int) * definedSize * definedSize, cudaMemcpyDeviceToHost);
    cudaFree(dPoints);

    // create random colors:
    cout << "Generating triangle colors...\n";
    float* colors = (float*)malloc(sizeof(float) * 3 * validTriangles);
    for(int i = 0; i < validTriangles * 3; i++){
        colors[i] = static_cast <float> (rand() / static_cast <float> (RAND_MAX));
    }

    auto end = chrono::steady_clock::now();
	auto timePassed = chrono::duration_cast<std::chrono::microseconds>(end - start);

    // write
    cout << "Rasterization completed after " << (timePassed.count() / 1000) << "ms!\nWriting results...\n";
    float* pixel = (float*)malloc(sizeof(float) * 3);
    readyOutputFile(fileName, (timePassed.count() / 1000));
    for (int i = 0; i < definedSize * definedSize; i++){
        if (pointTests[i] <= 0) {
            pixel[0] = pixel[1] = pixel[2] = 0;
        }
        else {
            // do color
            pixel[0] = colors[(pointTests[i] - 1)*3];
            pixel[1] = colors[(pointTests[i] - 1)*3 + 1];
            pixel[2] = colors[(pointTests[i] - 1)*3 + 2];
        }
        writeVertex(fileName, pixel);
    }
    free(pixel);

    cout << "All done!" << endl;
    auto realEnd = chrono::steady_clock::now();
	timePassed = chrono::duration_cast<std::chrono::microseconds>(realEnd - start);
    cout << "Complete process took " << (timePassed.count() / 1000) << "ms\n";

    return 0;
}

// Leah - how to run in command line:
// compile: g++ -o rasterize.exe rasterize.cpp filehandler.cpp barycentric.cpp
// execute: read.exe [fileName no extention]