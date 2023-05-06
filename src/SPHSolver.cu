#include "SPHSolver.cuh"
#include "SPHKernels.cuh"
#include "stdio.h"
#include "types.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <climits>
#include <glm/gtx/color_space.hpp>
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>

__constant__ glm::ivec3 NEIGH_DISPLACEMENTS[27];

#define forall_fluid_neighbors(code)\
    for (int neighDispIdx = 0; neighDispIdx < 27; ++neighDispIdx)\
    {\
        glm::ivec3 neighborCell = cellIdx + NEIGH_DISPLACEMENTS[neighDispIdx];\
        uint32_t neighborHashedCell = getHashedCell(neighborCell, size * 2);\
        uint32_t j = cellOffsetBuffer[neighborHashedCell];\
        while(j < fluidSize && cellIndexBuffer[j] == neighborHashedCell)\
        {\
            code\
            j++;\
        }\
    }\

#define forall_boundary_neighbors(code)\
    for (int neighDispIdx = 0; neighDispIdx < 27; ++neighDispIdx)\
    {\
        glm::ivec3 neighborCell = cellIdx + NEIGH_DISPLACEMENTS[neighDispIdx];\
        uint32_t neighborHashedCell = getHashedCell(neighborCell, size * 2);\
        uint32_t b = cellOffsetBoundary[neighborHashedCell];\
        while(b < size && cellIndexBuffer[b] == neighborHashedCell)\
        {\
            code\
            b++;\
        }\
    }\

__device__ glm::vec4 colorById(uint32_t id)
{
    curandState_t state;
    curand_init(id, 0, 0, &state); 
    return glm::vec4(
        curand_uniform(&state),
        curand_uniform(&state),
        curand_uniform(&state),
        1.0f
    );
}

__device__ glm::vec4 colorBySpeed(float normSpeed)
{
    glm::vec3 hsv = glm::mix(glm::vec3(210.0f, 1.0f, 1.0f), glm::vec3(210.0f, 0.13f, 1.0), normSpeed);
    return glm::vec4(glm::rgbColor(hsv), 1.0f);
}

// __device__ __host__ uint32_t getHashedCell(const glm::ivec3 key, const uint32_t size) 
// {
//     const uint32_t seed = 0xDEADBEEF;
//     uint32_t hash = seed;
//     const uint32_t prime = 16777619u;
//     const uint32_t x = static_cast<uint32_t>(key.x);
//     const uint32_t y = static_cast<uint32_t>(key.y);
//     const uint32_t z = static_cast<uint32_t>(key.z);
    
//     hash ^= x + seed;
//     hash *= prime;
//     hash ^= y + seed;
//     hash *= prime;
//     hash ^= z + seed;
//     hash *= prime;
    
//     return hash % size;
// }

__host__ __device__ uint32_t getHashedCell(const glm::ivec3 key, const uint32_t size)
{
    const uint32_t seed = 0xDEADBEEF;
    uint32_t hash = seed;
    const uint32_t prime = 16777619u;
    const uint32_t x = static_cast<uint32_t>(key.x);
    const uint32_t y = static_cast<uint32_t>(key.y);
    const uint32_t z = static_cast<uint32_t>(key.z);

    // z order / morton encoding
    uint32_t morton = 0;
    for (int i = 0; i < 10; ++i) {
        morton |= ((x & (1 << i)) << (2 * i)) |
                  ((y & (1 << i)) << (2 * i + 1)) |
                  ((z & (1 << i)) << (2 * i + 2));
    }

    hash ^= morton + seed;
    hash *= prime;

    return hash % size;
}


__global__ void computeDensity(glm::vec4* positions, float* densities, float* pressures, float* volumes, uint32_t* cellIndexBuffer, uint32_t* cellOffsetBuffer, uint32_t* cellOffsetBoundary, float* h_ptr, float* mass_ptr, float* density0_ptr, float* cubicConst_ptr, float* stiffness_ptr, int* boundarySize_ptr, int* fluidSize_ptr)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int fluidSize = *fluidSize_ptr;
    int size = fluidSize + *boundarySize_ptr;

    if (i < fluidSize)
    {
        float h = *h_ptr;
        float mass = *mass_ptr;
        float cubicConst = *cubicConst_ptr;
        float density0 = *density0_ptr;
        float stiffness = *stiffness_ptr;
        glm::vec4 ri = positions[i];
        glm::ivec3 cellIdx = glm::floor(glm::vec3(ri) / h);

        float density = 0.0f;
        forall_fluid_neighbors
        (
            glm::vec3 rij = glm::vec3(ri - positions[j]);
            density += cubicW(rij, h, cubicConst);
        );

        float bdensity = 0.0f;
        forall_boundary_neighbors
        (
            glm::vec3 rib = glm::vec3(ri - positions[b]);
            bdensity += volumes[b - fluidSize] * cubicW(rib, h, cubicConst);
        );

        density = (mass * density) + (density0 * bdensity);
        densities[i] = density;
        pressures[i] = max(0.0f, stiffness * (density - density0));
    }
}

__global__ void computeForces(glm::vec4* positions, glm::vec3* forces, glm::vec3* velocities, float* densities, float* pressures, float* volumes, uint32_t* cellIndexBuffer, uint32_t* cellOffsetBuffer, uint32_t* cellOffsetBoundary, float* density0_ptr, float* h_ptr, float* mass_ptr, float* spikyConst_ptr, float* viscosity_ptr, int* boundarySize_ptr, int* fluidSize_ptr)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int fluidSize = *fluidSize_ptr;
    int size = fluidSize + *boundarySize_ptr;

    if (i < fluidSize)
    {
        float density0 = *density0_ptr;
        float h = *h_ptr;
        float mass = *mass_ptr;
        float spikyConst = *spikyConst_ptr;
        float visco = *viscosity_ptr;
        float pi = pressures[i];
        float di = densities[i];
        glm::vec4 ri = positions[i];
        glm::vec3 vi = velocities[i];

        glm::vec3 pressForce = glm::vec3(0.0f);
        glm::vec3 viscoForce = glm::vec3(0.0f);
        glm::vec3 bPressForce = glm::vec3(0.0f);
        glm::vec3 bViscoForce = glm::vec3(0.0f);
        glm::ivec3 cellIdx = glm::floor(glm::vec3(ri) / h);
        float press_dens2 = pi / (di * di);

        forall_fluid_neighbors
        (
            glm::vec3 rij = glm::vec3(ri - positions[j]);

            pressForce -= (pressures[j] / (densities[j] * densities[j]) + press_dens2) * spikyW(rij, h, spikyConst);
            viscoForce += (velocities[j] - vi) / densities[j] * laplW(rij, h, spikyConst);
        );

        forall_boundary_neighbors
        (
            glm::vec3 rib = glm::vec3(ri - positions[b]);

            bPressForce -= volumes[b - fluidSize] * spikyW(rib, h, spikyConst); 
            bViscoForce += volumes[b - fluidSize] * laplW(rib, h, spikyConst);
        );

        bPressForce = bPressForce * (press_dens2 * 2.0f);
        bViscoForce = bViscoForce * ( (- vi) / di) * visco * 0.1f;
        viscoForce = viscoForce * visco;
        forces[i] += (viscoForce + pressForce) * mass + (bViscoForce + bPressForce) * density0;
    }
}

__global__ void integration(glm::vec4* positions, glm::vec3* forces, glm::vec3* velocities, glm::vec4* colors, float* timeStep_ptr, int* size_ptr)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int size = *size_ptr;

    if (i < size)
    {
        float timeStep = *timeStep_ptr;
        const glm::vec3 G = glm::vec3(0, -9.8f, 0);

        velocities[i] += timeStep * (forces[i] + G);
        positions[i] += glm::vec4(timeStep * velocities[i], 0.0f);

        forces[i] = glm::vec3(0.0f);

        // Color por velocidad
        float speed = glm::length(velocities[i]);
        const float maxSpeed = 4.0f; 
        colors[i] = colorBySpeed(speed / maxSpeed);
    }
}

__global__ void simpleBoundaryCondition(glm::vec4* positions, glm::vec3* velocities, glm::vec3* min_ptr, glm::vec3* max_ptr, int* size_ptr)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int size = *size_ptr;

    if (i < size)
    {
        glm::vec3 pos = glm::vec3(positions[i]);
        glm::vec3 vel = glm::vec3(velocities[i]);
        glm::vec3 min = *min_ptr;
        glm::vec3 max = *max_ptr;
        float restitution = 0.01f;

        if (pos.x < min.x)
        {
            pos.x = min.x;
            vel.x *= - restitution;
        }
        else if (pos.x > max.x)
        {
            pos.x = max.x;
            vel.x *= - restitution;
        }

        if (pos.y < min.y)
        {
            pos.y = min.y;
            vel.y *= - restitution;
        }
        else if (pos.y > max.y)
        {
            pos.y = max.y;
            vel.y *= - restitution;
        }

        if (pos.z < min.z)
        {
            pos.z = min.z;
            vel.z *= - restitution;
        }
        else if (pos.z > max.z)
        {
            pos.z = max.z;
            vel.z *= - restitution;
        }

        positions[i] = glm::vec4(pos, 1.0f);
        velocities[i] = vel;
    }
}

__global__ void resetCuda(glm::vec3* velocities, glm::vec3* forces, int* size_ptr)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int size = *size_ptr;

    if (i < size)
    {
        velocities[i] = glm::vec3(0.0f);
        forces[i] = glm::vec3(0.0f);
    }
}

__global__ void resetCellOffset(uint32_t* cellOffset, int* size_ptr)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int size = *size_ptr * 2;

    if (i < size)
    {
        cellOffset[i] = size;
    }
}

__global__ void insertParticles(glm::vec4* positions, uint32_t* cellIndexBuffer, glm::vec4* colors, float* h_ptr, int* boundarySize_ptr, int* fluidSize_ptr)
{
    int fluidSize = *fluidSize_ptr;
    int size = *boundarySize_ptr + fluidSize;
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < fluidSize)
    {
        float h = *h_ptr;

        glm::vec3 pos = glm::vec3(positions[i]);

        glm::ivec3 cell = glm::floor(pos / h);
        uint32_t hashedCell = getHashedCell(cell, size * 2);
        cellIndexBuffer[i] = hashedCell;

        // Color por celda
        //colors[i] = colorById(hashedCell);
    }
}

__global__ void insertBoundaryParticles(glm::vec4* positions, uint32_t* cellIndexBuffer, glm::vec4* colors, float* h_ptr, int* boundarySize_ptr, int* fluidSize_ptr)
{
    int boundarySize = *boundarySize_ptr;
    int fluidSize = *fluidSize_ptr;
    int size = boundarySize + fluidSize;
    int i = blockDim.x * blockIdx.x + threadIdx.x + fluidSize;

    if (i < size)
    {
        float h = *h_ptr;

        glm::vec3 pos = glm::vec3(positions[i]);

        glm::ivec3 cell = glm::floor(pos / h);
        uint32_t hashedCell = getHashedCell(cell, size * 2);
        cellIndexBuffer[i] = hashedCell;

        // Color por celda
        //colors[i] = colorById(hashedCell);
    }
}

__global__ void computeCellOffset(uint32_t* cellIndexBuffer, uint32_t* cellOffsetBuffer, int* size_ptr)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int size = *size_ptr;

    if (i < size)
    {
        uint32_t hashedCell = cellIndexBuffer[i];
        atomicMin(&cellOffsetBuffer[hashedCell], uint32_t(i));
    }
}

__global__ void computeBoundaryCellOffset(uint32_t* cellIndexBuffer, uint32_t* cellOffsetBuffer, int* boundarySize_ptr, int* fluidSize_ptr)
{
    int boundarySize = *boundarySize_ptr;
    int fluidSize = *fluidSize_ptr;
    int size = boundarySize + fluidSize;
    int i = blockDim.x * blockIdx.x + threadIdx.x + fluidSize;

    if (i < size)
    {
        uint32_t hashedCell = cellIndexBuffer[i];
        atomicMin(&cellOffsetBuffer[hashedCell], uint32_t(i));
    }
}

__global__ void computeVolume(glm::vec4* positions, float* volumes, uint32_t* cellIndexBuffer, uint32_t* cellOffsetBoundary, float* h_ptr, float* cubicConst_ptr, int* boundarySize_ptr, int* fluidSize_ptr)
{
    int boundarySize = *boundarySize_ptr;
    int fluidSize = *fluidSize_ptr;
    int size = boundarySize + fluidSize;
    int i = blockDim.x * blockIdx.x + threadIdx.x + fluidSize;

    if (i < size)
    {
        float cubicConst = *cubicConst_ptr;
        float h = *h_ptr;
        glm::vec4 ri = positions[i];
        glm::ivec3 cellIdx = glm::floor(glm::vec3(ri) / h);

        float delta = 0.0f;

        forall_boundary_neighbors
        (
            glm::vec3 rib = glm::vec3(ri - positions[b]);
            delta += cubicW(rib, h, cubicConst);
        )

        volumes[i - fluidSize] = 1.0f / delta; 
    }
}

void SPHSolver::step(VAO_t vao)
{
    glm::vec4* d_positions;
    glm::vec4* d_colors;

    // Map resources (positions and colors)
    checkCudaErrors(cudaGraphicsMapResources(1, &vao.cuda_p_id, 0)); 
    checkCudaErrors(cudaGraphicsMapResources(1, &vao.cuda_c_id, 0)); 
    // Get pointers of mapped data (positions and colors)
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_positions, nullptr, vao.cuda_p_id)); 
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_colors, nullptr, vao.cuda_c_id)); 

    dim3 blockDim(numThreadsPerBlock);
    dim3 cellGridDim((unsigned) ceil(2 * *size / (float) numThreadsPerBlock));
    dim3 gridDim((unsigned) ceil(*fluidSize / (float) numThreadsPerBlock));

    // CALCULO DE VECINOS
    resetCellOffset<<<cellGridDim, blockDim>>>(d_cellOffsetBuffer, d_size);
    checkCudaErrors(cudaGetLastError());
    
    insertParticles<<<gridDim, blockDim>>>(d_positions, d_cellIndexBuffer, d_colors, d_h, d_boundarySize, d_fluidSize);
    checkCudaErrors(cudaGetLastError());

    // Sort cellIndexBuffer, positions and velocities
    thrust::device_ptr<uint32_t> cellIndexPtr = thrust::device_pointer_cast(d_cellIndexBuffer);
    thrust::device_ptr<glm::vec4> posPtr = thrust::device_pointer_cast(d_positions);
    thrust::device_ptr<glm::vec3> velPtr = thrust::device_pointer_cast(d_velocities);
    thrust::sort_by_key(cellIndexPtr, cellIndexPtr + (*fluidSize), thrust::make_zip_iterator(thrust::make_tuple(posPtr, velPtr)));

    computeCellOffset<<<gridDim, blockDim>>>(d_cellIndexBuffer, d_cellOffsetBuffer, d_fluidSize);
    checkCudaErrors(cudaGetLastError());

    // CALCULO DE DENSIDAD Y PRESION
    computeDensity<<<gridDim, blockDim>>>(d_positions, d_densities, d_pressures, d_volumes, d_cellIndexBuffer, d_cellOffsetBuffer, d_cellOffsetBoundary, d_h, d_mass, d_density0, d_cubicConstK, d_stiffness, d_boundarySize, d_fluidSize);
    checkCudaErrors(cudaGetLastError());

    // CALCULO DE FUERZAS
    computeForces<<<gridDim, blockDim>>>(d_positions, d_forces, d_velocities, d_densities, d_pressures, d_volumes, d_cellIndexBuffer, d_cellOffsetBuffer, d_cellOffsetBoundary, d_density0, d_h, d_mass, d_spikyConst, d_viscosity, d_boundarySize, d_fluidSize);
    checkCudaErrors(cudaGetLastError());

    // INTEGRACION NUMERICA
    integration<<<gridDim, blockDim>>>(d_positions, d_forces, d_velocities, d_colors, d_timeStep, d_fluidSize);
    checkCudaErrors(cudaGetLastError());

    //simpleBoundaryCondition<<<gridDim, blockDim>>>(d_positions, d_velocities, d_minDomain, d_maxDomain, d_fluidSize);
    //checkCudaErrors(cudaGetLastError());

    // Unmap resources
    checkCudaErrors(cudaGraphicsUnmapResources(1, &vao.cuda_p_id, 0)); 
    checkCudaErrors(cudaGraphicsUnmapResources(1, &vao.cuda_c_id, 0)); 
}

void SPHSolver::init()
{
    float h = *this->h;
    int boundarySize = *this->boundarySize; 
    int size = *this->size;
    
    // Variables que no existen fuera de la clase SPHSolver y por lo tanto se reserva memoria para ellas
    this->spikyConst = new float(45.0f / ((float) (M_PI * pow(h, 6.0))));
    this->cubicConstK = new float(8.0f / ((float) M_PI * h * h * h));   
    this->cellIndexBuffer = new uint32_t[size]; 
    this->cellOffsetBuffer = new uint32_t[2 * size];
    this->cellOffsetBoundary = new uint32_t[2 * size];
    this->volumes = new float[boundarySize];

    for (int i = 0; i < boundarySize; ++i)
    {
        volumes[i] = 0.0f;
    }  

    allocateCudaMemory();
  
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    if (nDevices <= 0) 
    {
        std::cerr << "Any GPU device found" << std::endl;
        exit(1);
    }
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    if (props.maxThreadsPerBlock < numThreadsPerBlock)
        numThreadsPerBlock = props.maxThreadsPerBlock;
}

void SPHSolver::reset(VAO_t vao, glm::vec4* h_positions)
{
    size_t bytes;
    glm::vec4* d_positions;

    // Map resources
    checkCudaErrors(cudaGraphicsMapResources(1, &vao.cuda_p_id, 0));
    // Get pointer of mapped data
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_positions, &bytes, vao.cuda_p_id)); 

    dim3 blockDim(numThreadsPerBlock);
    dim3 gridDim((unsigned) ceil(*size / (float) numThreadsPerBlock));
    resetCuda<<<gridDim, blockDim>>>(d_velocities, d_forces, d_size);
    checkCudaErrors(cudaGetLastError());

    // Unmap resources
    checkCudaErrors(cudaGraphicsUnmapResources(1, &vao.cuda_p_id, 0)); 

    // Copy positions
    checkCudaErrors(cudaMemcpy(d_positions, h_positions, sizeof(glm::vec4) * *size, cudaMemcpyHostToDevice));

    computeBoundaryNeighbors(vao);
}

void SPHSolver::computeBoundaryNeighbors(VAO_t vao)
{
    glm::vec4* d_positions;
    glm::vec4* d_colors;

    // Map resources (positions and colors)
    checkCudaErrors(cudaGraphicsMapResources(1, &vao.cuda_p_id, 0)); 
    checkCudaErrors(cudaGraphicsMapResources(1, &vao.cuda_c_id, 0)); 
    // Get pointers of mapped data (positions and colors)
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_positions, nullptr, vao.cuda_p_id)); 
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_colors, nullptr, vao.cuda_c_id)); 

    dim3 blockDim(numThreadsPerBlock);
    dim3 gridDim((unsigned) ceil(2 * *size / (float) numThreadsPerBlock));

    // Reset boundary cell offsets
    resetCellOffset<<<gridDim, blockDim>>>(d_cellOffsetBoundary, d_size);
    checkCudaErrors(cudaGetLastError());

    gridDim = dim3((unsigned) ceil(*boundarySize / (float) numThreadsPerBlock));
    
    // Insert boundary particles 
    insertBoundaryParticles<<<gridDim, blockDim>>>(d_positions, d_cellIndexBuffer, d_colors, d_h, d_boundarySize, d_fluidSize);
    checkCudaErrors(cudaGetLastError());

    // Ordenar cellIndexBuffer, positions y velocities (La parte de los arrays correspondiente a las boundary particles [fluidSize, size])
    thrust::device_ptr<uint32_t> cellIndexPtr = thrust::device_pointer_cast(d_cellIndexBuffer + *fluidSize);
    thrust::device_ptr<glm::vec4> posPtr = thrust::device_pointer_cast(d_positions + *fluidSize);
    thrust::device_ptr<glm::vec3> velPtr = thrust::device_pointer_cast(d_velocities + *fluidSize);
    thrust::sort_by_key(cellIndexPtr, cellIndexPtr + (*boundarySize), thrust::make_zip_iterator(thrust::make_tuple(posPtr, velPtr)));

    // Cell offsets
    computeBoundaryCellOffset<<<gridDim, blockDim>>>(d_cellIndexBuffer, d_cellOffsetBoundary, d_boundarySize, d_fluidSize);
    checkCudaErrors(cudaGetLastError());

    // Compute volume
    computeVolume<<<gridDim, blockDim>>>(d_positions, d_volumes, d_cellIndexBuffer, d_cellOffsetBoundary, d_h, d_cubicConstK, d_boundarySize, d_fluidSize);
    checkCudaErrors(cudaGetLastError());

    // Unmap resources
    checkCudaErrors(cudaGraphicsUnmapResources(1, &vao.cuda_p_id, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &vao.cuda_c_id, 0)); 
}

void SPHSolver::allocateCudaMemory()
{
    // Reservar memoria en el device
    checkCudaErrors(cudaMalloc(&d_h, sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_timeStep, sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_cubicConstK, sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_spikyConst, sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_radius, sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_mass, sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_density0, sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_stiffness, sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_viscosity, sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_size, sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_fluidSize, sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_boundarySize, sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_densities, sizeof(float) * *fluidSize));
    checkCudaErrors(cudaMalloc(&d_pressures, sizeof(float) * *fluidSize));
    checkCudaErrors(cudaMalloc(&d_forces, sizeof(glm::vec3) * *fluidSize));
    checkCudaErrors(cudaMalloc(&d_velocities, sizeof(glm::vec3) * *size));
    checkCudaErrors(cudaMalloc(&d_minDomain, sizeof(glm::vec3)));
    checkCudaErrors(cudaMalloc(&d_maxDomain, sizeof(glm::vec3)));
    checkCudaErrors(cudaMalloc(&d_cellIndexBuffer, sizeof(uint32_t) * (*size)));
    checkCudaErrors(cudaMalloc(&d_cellOffsetBuffer, sizeof(uint32_t) * (*size * 2)));
    checkCudaErrors(cudaMalloc(&d_cellOffsetBoundary, sizeof(uint32_t) * (*size * 2)));
    checkCudaErrors(cudaMalloc(&d_volumes, sizeof(float) * (*boundarySize)));

    // Copiar datos al device
    checkCudaErrors(cudaMemcpy(d_h, h, sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_timeStep, timeStep, sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_cubicConstK, cubicConstK, sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spikyConst, spikyConst, sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_radius, radius, sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_mass, mass, sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_density0, density0, sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_stiffness, stiffness, sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_viscosity, viscosity, sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_size, size, sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fluidSize, fluidSize, sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_boundarySize, boundarySize, sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_densities, densities, sizeof(float) * (*fluidSize), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_pressures, pressures, sizeof(float) * (*fluidSize), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_forces, forces, sizeof(float) * (*fluidSize), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_velocities, velocities, sizeof(float) * (*fluidSize), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_minDomain, minDomain, sizeof(glm::vec3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_maxDomain, maxDomain, sizeof(glm::vec3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_cellIndexBuffer, cellIndexBuffer, sizeof(uint32_t) * (*size), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_cellOffsetBuffer, cellOffsetBuffer, sizeof(uint32_t) * (*size * 2), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_cellOffsetBoundary, cellOffsetBoundary, sizeof(uint32_t) * (*size * 2), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_volumes, volumes, sizeof(float) * (*boundarySize), cudaMemcpyHostToDevice));

    // Copy constant symbol
    std::vector<glm::ivec3> displacements = { glm::ivec3(-1, -1, -1), glm::ivec3(-1, -1, 0), glm::ivec3(-1, -1, 1), glm::ivec3(-1, 0, -1), glm::ivec3(-1, 0, 0), glm::ivec3(-1, 0, 1), glm::ivec3(-1, 1, -1), glm::ivec3(-1, 1, 0), glm::ivec3(-1, 1, 1), glm::ivec3(0, -1, -1), glm::ivec3(0, -1, 0), glm::ivec3(0, -1, 1), glm::ivec3(0, 0, -1), glm::ivec3(0, 0, 0), glm::ivec3(0, 0, 1), glm::ivec3(0, 1, -1), glm::ivec3(0, 1, 0), glm::ivec3(0, 1, 1), glm::ivec3(1, -1, -1), glm::ivec3(1, -1, 0), glm::ivec3(1, -1, 1), glm::ivec3(1, 0, -1), glm::ivec3(1, 0, 0), glm::ivec3(1, 0, 1), glm::ivec3(1, 1, -1), glm::ivec3(1, 1, 0), glm::ivec3(1, 1, 1)};
    std::sort(displacements.begin(), displacements.end(), [&](glm::ivec3 a, glm::ivec3 b) {
        uint32_t hash_a = getHashedCell(a, *size * 2);
        uint32_t hash_b = getHashedCell(b, *size * 2);
        return hash_a < hash_b;
    });

    cudaMemcpyToSymbol(NEIGH_DISPLACEMENTS, displacements.data(), sizeof(glm::ivec3) * 27);
}

void SPHSolver::freeCudaMemory()
{
    checkCudaErrors(cudaFree(d_h));
    checkCudaErrors(cudaFree(d_timeStep));
    checkCudaErrors(cudaFree(d_cubicConstK));
    checkCudaErrors(cudaFree(d_spikyConst));
    checkCudaErrors(cudaFree(d_radius));
    checkCudaErrors(cudaFree(d_mass));
    checkCudaErrors(cudaFree(d_density0));
    checkCudaErrors(cudaFree(d_stiffness));
    checkCudaErrors(cudaFree(d_viscosity));
    checkCudaErrors(cudaFree(d_size));
    checkCudaErrors(cudaFree(d_fluidSize));
    checkCudaErrors(cudaFree(d_boundarySize));
    checkCudaErrors(cudaFree(d_densities));
    checkCudaErrors(cudaFree(d_pressures));
    checkCudaErrors(cudaFree(d_forces));
    checkCudaErrors(cudaFree(d_velocities));
    checkCudaErrors(cudaFree(d_minDomain));
    checkCudaErrors(cudaFree(d_maxDomain));
    checkCudaErrors(cudaFree(d_cellIndexBuffer));
    checkCudaErrors(cudaFree(d_cellOffsetBuffer));
    checkCudaErrors(cudaFree(d_cellOffsetBoundary));
    checkCudaErrors(cudaFree(d_volumes));
}

void SPHSolver::release()
{
    delete spikyConst;
    delete cubicConstK;
    delete[] cellIndexBuffer;
    delete[] cellOffsetBuffer;
    delete[] cellOffsetBoundary;
    delete[] volumes;

    freeCudaMemory();
}

