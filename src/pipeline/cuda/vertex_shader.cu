#include "pipeline/cuda/vertex_shader.cuh"

__global__ void perspectiveKernel(const cuda_vec4* in, float* out, int n,
                              const float* proj4x4,
                              float cam_x, float cam_y, float cam_z) {
    // calculate thread index
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) return;
    cuda_vec4 v = in[idx];

    // apply camera offset
    v.vx -= cam_x; v.vy -= cam_y; v.vz -= cam_z;
    float x = proj4x4[0]*v.vx + proj4x4[4]*v.vy + proj4x4[8]*v.vz + proj4x4[12];
    float y = proj4x4[1]*v.vx + proj4x4[5]*v.vy + proj4x4[9]*v.vz + proj4x4[13];
    float w = proj4x4[3]*v.vx + proj4x4[7]*v.vy + proj4x4[11]*v.vz + proj4x4[15];
    if (w>0.0f) {
        out[2*idx+0] = x/w;
        out[2*idx+1] = y/w;
    } else {
        out[2*idx+0] = 0.0f;
        out[2*idx+1] = 0.0f;
    }
}
