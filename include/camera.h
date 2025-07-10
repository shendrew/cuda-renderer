#ifndef CAMERA_H
#define CAMERA_H

#include "matrix.h"
#include "common.h"

struct Camera {
    Camera(float x, float y, float z, float fov);

    Vec4 position;
    // Vec4 rotation;
    float fov;
    Matrix perspectiveMat;
};

#endif  //CAMERA_H