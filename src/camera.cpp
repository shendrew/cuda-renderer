#include "camera.h"

Camera::Camera(float x, float y, float z, float fov)
: position({x, y, z, 0}), fov(fov), perspectiveMat(Matrix(4, 4)) {
    const float NEAR = 1.0f;
    const float FAR = 100.0f;
    const float FOV = fov * M_PI / 180.0;     // degree to radian
    const float aspect = (float)WINDOW_WIDTH / (float)WINDOW_HEIGHT;

    float scale = 1 / tan(FOV / 2);
    float scale_x = scale / aspect;
    float scale_y = scale;
    float A = -(FAR + NEAR) / (FAR - NEAR);
    float B = -(2 * FAR * NEAR) / (FAR - NEAR);

    perspectiveMat = Matrix({
        {scale_x, 0, 0, 0},
        {0, scale_y, 0, 0},
        {0, 0, A, B},
        {0, 0, -1.0, 0}
    });
}
