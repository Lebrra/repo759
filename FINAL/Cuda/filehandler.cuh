#ifndef FILEHANDLER_CUH
#define FILEHANDLER_CUH

#include <string>
using namespace std;

// reads the first line of [fileName]_vertices.txt generated from stlToTxt.py
__host__ void getVertexCount(string fileName, int* count);

// reads in a list of x, y, z vertices from a file named [fileName]_vertices.txt generated from stlToTxt.py
__host__ void readVertices(string fileName, float* vertices);

// reads the first line of [fileName]_faces.txt generated from stlToTxt.py
__host__ void getFaceCount(string fileName, int* count);

// reads in a list of pairs of 3 indices that form a triangle from a file named [fileName]_faces.txt generated from stlToTxt.py
__host__ void readFaces(string fileName, int* faces);

// prepare file [fileName]_output.txt for writing
__host__ void readyOutputFile(string fileName, long time);

// writes a vertex into the output file [fileName]_output.txt (vertex = x, y, z)
__host__ void writeVertex(string fileName, float* vertex);

#endif