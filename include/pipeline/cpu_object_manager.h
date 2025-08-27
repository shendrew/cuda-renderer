#ifndef CPU_OBJECT_MANAGER_H
#define CPU_OBJECT_MANAGER_H

#include "object_manager.h"

struct CPUObjectManager : ObjectManager {
    CPUObjectManager(const std::vector<Vec4>& verts, Camera& cam, uint32_t vbo, uint32_t vao);
    void render() override;
private:
    std::vector<Vec4>   m_vertices;
    Camera&             m_camera;
    uint32_t            m_vbo;
    uint32_t            m_vao;
};

#endif // CPU_OBJECT_MANAGER_H