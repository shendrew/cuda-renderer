#ifndef CUDA_OBJECT_MANAGER_H
#define CUDA_OBJECT_MANAGER_H

#include "object_manager.h"
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

struct CUDAObjectManager : ObjectManager {
    CUDAObjectManager(const std::vector<Vec4>& verts, Camera* cam, uint32_t vbo);
    ~CUDAObjectManager();
    void render() override;
private:
    Vec4*                   d_vertices;
    size_t                  m_count;
    uint32_t                m_vao;
    cudaGraphicsResource*   m_cudaVbo;
    Camera*                 m_camera;
    Matrix                  m_projMat;
    int                     m_threadCnt = 128;
};

#endif // CUDA_OBJECT_MANAGER_H