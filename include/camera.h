#ifndef CAMERA_H
#define CAMERA_H

#include "./geometry.h"

class Camera {
public:
    Camera(float x, float y, float z);

    Vec4 position;
    Vec4 rotation;

    //! want to use matrix directly
    // Vec3 position;
    // Quarternion rotation;
};

#endif  //CAMERA_H