#include "pipeline/cuda/vertex_shader.cuh"

using FuncPtr = void(*)(int *a, int *b);

__global__ void addVectorsKernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

extern "C" {
__global__ void projectKernel(const v4* in, float* out, int n,
                              const float* proj4x4,
                              float camx, float camy, float camz) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=n) return;
    v4 v = in[i];
    // apply camera offset
    v.vx -= camx; v.vy -= camy; v.vz -= camz;
    float x = proj4x4[0]*v.vx + proj4x4[4]*v.vy + proj4x4[8]*v.vz + proj4x4[12];
    float y = proj4x4[1]*v.vx + proj4x4[5]*v.vy + proj4x4[9]*v.vz + proj4x4[13];
    float w = proj4x4[3]*v.vx + proj4x4[7]*v.vy + proj4x4[11]*v.vz + proj4x4[15];
    if (w>0.0f) {
        out[2*i+0] = (x/w) * (WINDOW_WIDTH/2.0f);
        out[2*i+1] = (y/w) * (WINDOW_HEIGHT/2.0f);
    } else {
        out[2*i+0] = 0.0f; out[2*i+1] = 0.0f;
    }
}
}

// void processVerticesOnGPU(const std::vector<v4>& vertices, const float projectionMatrix[16], std::vector<vec4>& projectedVertices) {
//     // 1. Copy Data Once
//     Vec4* gpuVertices;
//     cudaMalloc(&gpuVertices, vertices.size() * sizeof(vec4));
//     cudaMemcpy(gpuVertices, vertices.data(), vertices.size() * sizeof(vec4), cudaMemcpyHostToDevice);

//     vec4* gpuProjectedVertices;
//     cudaMalloc(&gpuProjectedVertices, vertices.size() * sizeof(vec4));

//     // 2. CUDA Kernel
//     dim3 blockDim(256);
//     dim3 gridDim((vertices.size() + blockDim.x - 1) / blockDim.x);

//     // Kernel Launch
//     kernel<<<gridDim, blockDim>>>(gpuVertices, projectionMatrix, gpuProjectedVertices, vertices.size());

//     // 3. Data Reuse (Copy results back to CPU)
//     projectedVertices.resize(vertices.size());
//     cudaMemcpy(projectedVertices.data(), gpuProjectedVertices, vertices.size() * sizeof(vec4), cudaMemcpyDeviceToHost);

//     // Free GPU memory
//     cudaFree(gpuVertices);
//     cudaFree(gpuProjectedVertices);
// }