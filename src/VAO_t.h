#pragma once
#include "types.h"
#include "cuda_gl_interop.h"

struct VAO_t
{
    uint id;   // Id del vertex buffer object
    uint v_id; // Id del buffer de vertices
    uint i_id; // Id del buffer de indices
    uint mvp_id;
    cudaGraphicsResource* cuda_id = nullptr;
    //uint color_id;
    //uint tan_id;
};