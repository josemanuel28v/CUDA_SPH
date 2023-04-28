#pragma once
#define GLAD_ONLY_HEADERS
#include "common.h"
#include "cuda_gl_interop.h"
#include "types.h"

struct VAO_t
{
    uint id;   // Id del vertex buffer object
    uint v_id; // Id del buffer de vertices
    uint i_id; // Id del buffer de indices

    uint p_id; // Id del buffer de posiciones de cada particula
    uint c_id; // Id del buffer de color de cada particula
    
    cudaGraphicsResource* cuda_p_id = nullptr; // Id del buffer de posiciones para CUDA
    cudaGraphicsResource* cuda_c_id = nullptr; // Id del buffer de color para CUDA
};