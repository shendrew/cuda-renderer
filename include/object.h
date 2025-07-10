#ifndef OBJECT_H
#define OBJECT_H

#include "std_includes.h"
#include "matrix.h"
#include <sstream>

struct Object {
    Vec4 pos;
    std::vector<Vec4> vertices;
    std::vector<std::vector<int>> faces;

    Object();
    Object(float x, float y, float z);
};

std::istream& operator>>(std::istream& in, Object& obj);


#endif // OBJECT_H