// inspiration: https://github.com/ssloy/tinyrenderer/wiki/Lesson-2:-Triangle-rasterization-and-back-face-culling

#include <iostream>
#include <chrono>
#include "filehandler.h"
#include "pixel.h"
#include "omp.h"

// hardset output height and width
const int definedSize = 256;
const float padding = 10;

// vertices.length == vertexCount * 3 (1d array of 3d vertices)
void adjustToSize(float* vertices, int vertexCount) {
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
        multiplier = (definedSize - padding*2) / pointsWidth;
    }
    else { 
        multiplier = (definedSize - padding*2) / pointsHeight;
    }

    // apply multiplier to all points (and offset if any points are negative)
    #pragma omp parallel for
    for (int i = 0; i < vertexCount; i++) {
        int x = i * 3;  // x + 1 = y
        if (minX < 0){
            vertices[x] -= minX;
        }
        if (minY < 0){
           vertices[x + 1] -= minY; 
        }

        vertices[x] *= multiplier;
        vertices[x + 1] *= multiplier;

        vertices[x] += padding;
        vertices[x+1] += padding;
    }
}

int main(int argc, char** argv) {
    if (argc <= 1){
        cout << "Please include a filename (excluding the '_vertices.txt' or '_faces.txt') to rasterize!";
        return 0;
    }
    // else I could add optional size values, but let's just hardset it to 256...

    omp_set_num_threads(definedSize);
    #pragma omp.h parallel num_threads(definedSize)
	{
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

        readVertices(fileName, vertices);
        readFaces(fileName, faces);

        cout << "This file has " << vertCount << " vertices and " << triangleCount << " triangles!" << endl;

        cout << "Adjusting to size..." << endl;
        adjustToSize(vertices, vertCount);

        int* pointTests = (int*)malloc(sizeof(int) * definedSize * definedSize);
        #pragma omp parallel for
        for(int i = 0; i < definedSize*definedSize; i++) pointTests[i] = 0;

        // do math
        int validTriangles = 0;
        cout << "Comparing pixels with triangles..." << endl;
        //float* triangle = (float*)malloc(sizeof(float) * 6);
        #pragma omp parallel for
        for(int tri = 0; tri < triangleCount; tri++){
            float* triangle = (float*)malloc(sizeof(float) * 6);

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

            // do parallelism here
            #pragma omp parallel for collapse(2)
            for (int x = 0; x < definedSize; x++){
                for (int y = 0; y < definedSize; y++){
                    if (pointTests[y*definedSize + x] <= 0){
                        bool isIn = inTriangle(triangle, x, y);
                        if (isIn) pointTests[y*definedSize + x] = validTriangles;
                    } 
                }
            }
            free(triangle);
        }

        // create random colors:
        cout << "Generating triangle colors...\n";
        float* colors = (float*)malloc(sizeof(float) * 3 * validTriangles);
        #pragma omp parallel for
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

        free(vertices);
        free(faces);
    }

    return 0;
}

// Leah - how to run in command line:
// compile: g++ -o rasterize.exe rasterize.cpp filehandler.cpp barycentric.cpp
// execute: read.exe [fileName no extention]