#include "pipeline/cuda/cuda_bridge.cuh"
#include "pipeline/cuda/vertex_shader.cuh"

void projectPerspective(
    int blocks, int threadCnt,
    cuda_vec4* d_vertices, float* cuda_output, int count, float* perspectiveMat, float vx, float vy, float vz)
{
    perspectiveKernel<<<blocks, threadCnt>>>(
        d_vertices, cuda_output,
        count,
        perspectiveMat,
        vx,
        vy,
        vz
    );
}