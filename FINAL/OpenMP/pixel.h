#ifndef PIXEL_H
#define PIXEL_H

using namespace std;

// barycentric calculation between 3 2D points
float barycentric(float p1X, float p1Y, float p2X, float p2Y, float p3X, float p3Y);

// iterates through every triangle to see if this pixel is within any given triangle
bool inTriangle(float* triangle, int x, int y);

#endif