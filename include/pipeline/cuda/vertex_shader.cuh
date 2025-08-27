#ifndef VERTEX_SHADER_CUH
#define VERTEX_SHADER_CUH

#include <cuda_runtime.h>
#include "common.h"
#include "pipeline/cuda/types.cuh"

__global__ void perspectiveKernel(const cuda_vec4* in, float* out, int n,
                              const float* proj4x4,
                              float camx, float camy, float camz); 

#endif // VERTEX_SHADER_CUH