
#ifndef PIXEL_H

// first triangle: 
// - point 1 = (points[triangle[0]], points[triangle[0] + 1], points[triangle[0] + 2])
// - point 2 = (points[triangle[0 + 1]], points[triangle[0 + 1] + 1], points[triangle[0 + 1] + 2])
// - point 3 = (points[triangle[0 + 2]], points[triangle[0 + 2] + 1], points[triangle[0 + 2] + 2])

//https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
float barycentric(float p1X, float p1Y, float p2X, float p2Y, float p3X, float p3Y){
    return (p1X - p3X) * (p2Y - p3Y) - (p2X - p3X) * (p1Y - p3Y);
}

bool inTriangle(float* triangle, int x, int y){
    float b1 = barycentric(x, y, triangle[0], triangle[1], triangle[2], triangle[3]);
    float b2 = barycentric(x, y, triangle[2], triangle[3], triangle[4], triangle[5]);
    float b3 = barycentric(x, y, triangle[4], triangle[5], triangle[0], triangle[1]);

    bool neg = b1 <= 0 && b2 <= 0 && b3 <= 0;
    bool pos = b1 >= 0 && b2 >= 0 && b3 >= 0;

    return neg || pos;
}

#endif