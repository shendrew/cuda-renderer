#ifndef OBJECT_MANAGER_H
#define OBJECT_MANAGER_H

#include "camera.h"
#include "matrix.h"
#include "common.h"
#include "std_includes.h"

struct ObjectManager {
    virtual ~ObjectManager() = default;
    virtual void render() = 0;
};

#endif
