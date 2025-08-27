#ifndef CUDA_OBJECT_MANAGER_H
#define CUDA_OBJECT_MANAGER_H

#include "object_manager.h"
#include "pipeline/cuda/types.cuh"
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

struct CUDAObjectManager : ObjectManager {
    CUDAObjectManager(const std::vector<Vec4>& verts, Camera& cam, uint32_t vbo, uint32_t vao);
    ~CUDAObjectManager();
    void render() override;
private:
    // device data
    cudaGraphicsResource*   d_vbo;
    uint32_t                m_vao;
    int                     m_threadCnt = 128;
    size_t                  m_count;

    // render data
    cuda_vec4*              d_vertices;
    Camera&                 m_camera;
};

#endif // CUDA_OBJECT_MANAGER_H