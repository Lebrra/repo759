#include <iostream>
#include <fstream>
#include <string>
using namespace std;

#ifndef FILEHANDLER_CUH

__host__ void getVertexCount(string fileName, int* count){
    ifstream readFile(fileName + "_vertices.txt");
    string line;
    const string delim = ", ";

    getline(readFile, line);
    count[0] = stoi(line);
}

__host__ void readVertices(string fileName, float* vertices) {
    ifstream readFile(fileName + "_vertices.txt");
    string line;
    const string delim = ", ";

    getline(readFile, line);
    int vertCount = stoi(line);

    for (int i = 0; i < vertCount; i++){
        getline(readFile, line);

        // each line is a 3D coordinate: "x, y, z"
        vertices[3 * i] = stof(line.substr(0, line.find(delim)));
        line = line.substr(line.find(delim) + 2, line.length());
        vertices[3 * i + 1] = stof(line.substr(0, line.find(delim)));
        line = line.substr(line.find(delim) + 2, line.length());
        vertices[3 * i + 2] = stof(line);
    }
}

__host__ void getFaceCount(string fileName, int* count){
    ifstream readFile(fileName + "_faces.txt");
    string line;
    const string delim = ", ";

    getline(readFile, line);
    count[0] = stoi(line);
}

__host__ void readFaces(string fileName, int* faces) {
    ifstream readFile(fileName + "_faces.txt");
    string line;
    const string delim = ", ";

    getline(readFile, line);
    int faceCount = stoi(line);

    for (int i = 0; i < faceCount; i++){
        getline(readFile, line);

        // each line is 3 indices of a coordinate from vertices that make a triangle: "int, int, int"
        faces[3 * i] = stoi(line.substr(0, line.find(delim)));
        line = line.substr(line.find(delim) + 2, line.length());
        faces[3 * i + 1] = stoi(line.substr(0, line.find(delim)));
        line = line.substr(line.find(delim) + 2, line.length());
        faces[3 * i + 2] = stoi(line);
    }
}

__host__ void readyOutputFile(string fileName, long time){
    ofstream of;
    of.open(fileName + "_output.txt", ofstream::out | ofstream::trunc);
    of << time << endl;
    of.close();
}

__host__ void writeVertex(string fileName, float* vertex){
    fstream f;
    f.open(fileName + "_output.txt", ios::app);
    f << vertex[0] << ", " << vertex[1] << ", " << vertex[2] << endl;
    f.close();
}

#endif