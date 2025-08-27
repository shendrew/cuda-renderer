#ifndef CUDA_BRIDGE_CUH
#define CUDA_BRIDGE_CUH

#include "pipeline/cuda/vertex_shader.cuh"
#include "pipeline/cuda/types.cuh"

void projectPerspective(
    int blocks, int m_threadCnt,
    cuda_vec4* d_vertices, float* cuda_output, int m_count, float* perspectiveMat, float vx, float vy, float vz
);

#endif //CUDA_BRIDGE_CUH