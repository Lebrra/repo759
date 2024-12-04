// inspiration: https://github.com/ssloy/tinyrenderer/wiki/Lesson-2:-Triangle-rasterization-and-back-face-culling

#include <cuda.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include "pixel.cuh"
#include "sizeAdjuster.cuh"
#include "filehandler.cuh"

using namespace std;

// hardset output height and width
const int definedSize = 256;
const float padding = 10;

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
    cout << "Getting vertex count..." << endl;
    int vertCount = getVertexCount(fileName);
    cout << "Getting face count..." << endl;
    int triangleCount = getFaceCount(fileName);

    cout << "Alotting pointers..." << endl;
    float* vertices = (float*)malloc(sizeof(float) * vertCount * 3);
    int* faces = (int*)malloc(sizeof(int) * triangleCount * 3);
    float *dVerts;
    
    cout << "Reading all vertices..." << endl;
    readVertices(fileName, vertices);
    cout << "Reading all faces..." << endl;
    readFaces(fileName, faces);

    cout << "This file has " << vertCount << " vertices and " << triangleCount << " triangles!" << endl;

    cout << "Adjusting to size..." << endl;
    cudaMalloc((void**)&dVerts, sizeof(float) * vertCount*3);
    cudaMemcpy(dVerts, &vertices, sizeof(float) * vertCount*3, cudaMemcpyHostToDevice);

    adjustSize(dVerts, vertCount, definedSize, padding);
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