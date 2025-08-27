#include "pipeline/cuda_object_manager.h"
#include "pipeline/cuda/vertex_shader.cuh"
#include "pipeline/cuda/cuda_bridge.cuh"

CUDAObjectManager::CUDAObjectManager(const std::vector<Vec4>& verts, Camera& cam, uint32_t vbo, uint32_t vao)
    : m_count{verts.size()}
    , m_camera{cam}
    , m_vao{vao}
{
    // copy vertex array flattened to 1D
    cudaMalloc(&d_vertices, m_count*sizeof(cuda_vec4));

    std::vector<float> data;
    data.reserve(m_count*4);
    for (const auto& vec : verts) {
        data.insert(data.end(), vec.value_ptr(), vec.value_ptr() + 4);
    }
    cudaMemcpy(d_vertices, data.data(), m_count*sizeof(cuda_vec4), cudaMemcpyHostToDevice);
    
    // register VBO
    cudaGraphicsGLRegisterBuffer(&d_vbo, vbo, cudaGraphicsMapFlagsWriteDiscard);
}

CUDAObjectManager::~CUDAObjectManager() {
    cudaGraphicsUnregisterResource(d_vbo);
    cudaFree(d_vertices);
}

void CUDAObjectManager::render() {
    // lock VBO
    float* cuda_vbo;
    size_t mapped_size;
    cudaGraphicsMapResources(1, &d_vbo, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&cuda_vbo, &mapped_size, d_vbo);
    // launch kernel
    cudaDeviceSynchronize();
    int blocks=(m_count+m_threadCnt-1)/m_threadCnt;

    projectPerspective(
        blocks, m_threadCnt,
        d_vertices,
        cuda_vbo,
        m_count,
        m_camera.perspectiveMat.value_ptr(),
        m_camera.position.vx(),
        m_camera.position.vy(),
        m_camera.position.vz()
    );

    // release VBO lock
    cudaGraphicsUnmapResources(1, &d_vbo, 0);

    // draw
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // DO NOT need to rebind VBO (inplace update)

    glBindVertexArray(m_vao);
    glDrawArrays(GL_POINTS, 0, m_count);
    glBindVertexArray(0);
}
