#include "pipeline/cuda_object_manager.h"
#include "pipeline/cuda/vertex_shader.cuh"  // declares projectKernel


CUDAObjectManager::CUDAObjectManager(const std::vector<Vec4>& verts, Camera* cam, uint32_t vbo, uint32_t vao)
  : m_count(verts.size()), m_camera(cam), m_projMat( cam->perspectiveMat ), m_vao{vao} {
    // copy vertex array once
    cudaMalloc(&d_vertices, m_count*sizeof(Vec4));
    cudaMemcpy(d_vertices, verts.data(), m_count*sizeof(Vec4), cudaMemcpyHostToDevice);
    // register VBO
    cudaGraphicsGLRegisterBuffer(&m_cudaVbo, vbo, cudaGraphicsMapFlagsWriteDiscard);
}

CUDAObjectManager::~CUDAObjectManager() {
    cudaGraphicsUnregisterResource(m_cudaVbo);
    cudaFree(d_vertices);
}

void CUDAObjectManager::render() {
    // map VBO
    float* dptr;
    size_t sz;
    cudaGraphicsMapResources(1, &m_cudaVbo, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &sz, m_cudaVbo);
    // launch kernel
    cudaDeviceSynchronize();
    int blocks=(m_count+m_threadCnt-1)/m_threadCnt;
    projectKernel<<<blocks,m_threadCnt>>>(
        d_vertices, dptr,
        (int)m_count,
        m_projMat.value_ptr(),
        m_camera->position.vx(),
        m_camera->position.vy(),
        m_camera->position.vz()
    );
    cudaGraphicsUnmapResources(1, &m_cudaVbo, 0);
}
