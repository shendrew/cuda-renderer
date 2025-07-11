#ifndef VERTEX_SHADER_CUH
#define VERTEX_SHADER_CUH

#include <cuda_runtime.h>
#include "common.h"

#ifdef __cplusplus
extern "C" {
struct v4 { float vx,vy,vz,vw; };
__global__ void projectKernel(const v4* in, float* out, int n,
                              const float* proj4x4,
                              float camx, float camy, float camz); 
}
#endif // __cplusplus

#endif // VERTEX_SHADER_CUH