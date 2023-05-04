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

#define BLOCK_SIZE 256

__constant__ glm::ivec3 NEIGH_DISPLACEMENTS[27];
#define forall_fluid_neighbors_unsorted(code)\
    for (int neighDispIdx = 0; neighDispIdx < 27; ++neighDispIdx)\
    {\
        glm::ivec3 neighborIndex = cellIdx + NEIGH_DISPLACEMENTS[neighDispIdx];\
        uint32_t neighborHashedCell = getHashedCell(neighborIndex, size * 2);\
        uint32_t neighborIterator = cellOffsetBuffer[neighborHashedCell];\
        while(neighborIterator != (size * 2) && neighborIterator < size)\
        {\
            uint32_t j = particleIndexBuffer[neighborIterator];\
            if(cellIndexBuffer[j] != neighborHashedCell)\
            {\
                break;\
            }\
            code\
            neighborIterator++;\
        }\
    }\

#define forall_fluid_neighbors(code)\
    for (int neighDispIdx = 0; neighDispIdx < 27; ++neighDispIdx)\
    {\
        glm::ivec3 neighborIndex = cellIdx + NEIGH_DISPLACEMENTS[neighDispIdx];\
        uint32_t neighborHashedCell = getHashedCell(neighborIndex, size * 2);\
        uint32_t j = cellOffsetBuffer[neighborHashedCell];\
        while(j != (size * 2) && j < fluidSize)\
        {\
            if(cellIndexBuffer[j] != neighborHashedCell)\
            {\
                break;\
            }\
            code\
            j++;\
        }\
    }\

#define forall_boundary_neighbors(code)\
    for (int neighDispIdx = 0; neighDispIdx < 27; ++neighDispIdx)\
    {\
        glm::ivec3 neighborIndex = cellIdx + NEIGH_DISPLACEMENTS[neighDispIdx];\
        uint32_t neighborHashedCell = getHashedCell(neighborIndex, size * 2);\
        uint32_t b = cellOffsetBoundary[neighborHashedCell];\
        while(b != (size * 2) && b < size)\
        {\
            if(cellIndexBuffer[b] != neighborHashedCell)\
            {\
                break;\
            }\
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

// (80000 particulas y + 50 fps)
__device__ uint32_t getHashedCell3(glm::ivec3 cellIdx, uint32_t size)
{
    const uint32_t p1 = 73856093; 
    const uint32_t p2 = 19349663;
    const uint32_t p3 = 83492791;

    int n = p1 * cellIdx.x ^ p2 *cellIdx.y ^ p3 * cellIdx.z;
    n %= size;

    return n;
}

// No negativos (con 80000 particulas 71 fps)
__device__ uint32_t getHashedCell2(const glm::ivec3 key, const uint32_t size) 
{
    uint32_t hash = 2166136261u;
    const uint32_t prime = 16777619u;
    const uint32_t x = key.x;
    const uint32_t y = key.y;
    const uint32_t z = key.z;
    
    hash = (hash ^ x) * prime;
    hash = (hash ^ y) * prime;
    hash = (hash ^ z) * prime;
    
    return hash % size;
}

// Admite negativos (con 80000 particulas ~ 79 fps)
__device__ uint32_t getHashedCell(const glm::ivec3 key, const uint32_t size) 
{
    const uint32_t seed = 0xDEADBEEF;
    uint32_t hash = seed;
    const uint32_t prime = 16777619u;
    const uint32_t x = static_cast<uint32_t>(key.x);
    const uint32_t y = static_cast<uint32_t>(key.y);
    const uint32_t z = static_cast<uint32_t>(key.z);
    
    hash ^= x + seed;
    hash *= prime;
    hash ^= y + seed;
    hash *= prime;
    hash ^= z + seed;
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
            density += mass * cubicW(rij, h, cubicConst);
        );

        float bdensity = 0.0f;
        forall_boundary_neighbors
        (
            glm::vec3 rib = glm::vec3(ri - positions[b]);
            bdensity += density0 * volumes[b - fluidSize] * cubicW(rib, h, cubicConst); // El índice j sale de un array de tamaño fluidSize + boundarySize ya que volumes solo tiene tamaño boundarySize su indice sera j - fluidSize
        );

        //printf("fdens, bdens %f %f \n", density, bdensity);

        density = density + bdensity;
        densities[i] = density;
        pressures[i] = max(0.0f, stiffness * (density - density0));
    }
}

__global__ void computeDensityUnsorted(glm::vec4* positions, float* densities, float* pressures, uint32_t* particleIndexBuffer, uint32_t* cellIndexBuffer, uint32_t* cellOffsetBuffer, float* h_ptr, float* mass_ptr, float* density0_ptr, float* cubicConst_ptr, float* stiffness_ptr, int* size_ptr)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int size = *size_ptr;

    if (i < size)
    {
        float h = *h_ptr;
        float mass = *mass_ptr;
        float cubicConst = *cubicConst_ptr;
        float density0 = *density0_ptr;
        float stiffness = *stiffness_ptr;
        glm::vec4 ri = positions[i];
        glm::vec3 pos = glm::vec3(ri);

        float density = 0.0f;
        glm::ivec3 cellIdx = glm::floor(pos / h);

        forall_fluid_neighbors_unsorted
        (
            glm::vec3 rij = glm::vec3(ri - positions[j]);
            density += cubicW(rij, h, cubicConst);
        );

        density *= mass;
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

        glm::vec3 pforce = glm::vec3(0.0f);
        glm::vec3 vforce = glm::vec3(0.0f);
        glm::vec3 vbforce = glm::vec3(0.0f);
        glm::vec3 pos = glm::vec3(ri);
        glm::ivec3 cellIdx = glm::floor(pos / h);
        float press_dens2 = pi / (di * di);

        forall_fluid_neighbors
        (
            glm::vec3 rij = glm::vec3(ri - positions[j]);

            pforce -= mass * (pressures[j] / (densities[j] * densities[j]) + press_dens2) * spikyW(rij, h, spikyConst);
            vforce += mass * (velocities[j] - vi) / densities[j] * laplW(rij, h, spikyConst);
        );

        forall_boundary_neighbors
        (
            glm::vec3 rib = glm::vec3(ri - positions[b]);

            pforce -= density0 * volumes[b - fluidSize] * (press_dens2 * 2.0f) * spikyW(rib, h, spikyConst); // si no va bien probar con cubicW aunque en ese caso arriba tambien se debería cambiar
            vbforce += density0 * volumes[b - fluidSize] * (- vi) / di * laplW(rib, h, spikyConst);
        );

        forces[i] += (vforce * visco + vbforce * visco * 0.1f + pforce);
    }
}

__global__ void computeForcesUnsorted(glm::vec4* positions, glm::vec3* forces, glm::vec3* velocities, float* densities, float* pressures, uint32_t* particleIndexBuffer, uint32_t* cellIndexBuffer, uint32_t* cellOffsetBuffer, float* h_ptr, float* mass_ptr, float* spikyConst_ptr, float* viscosity_ptr, int* size_ptr)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int size = *size_ptr;

    if (i < size)
    {
        float h = *h_ptr;
        float mass = *mass_ptr;
        float spikyConst = *spikyConst_ptr;
        float visco = *viscosity_ptr;
        float pi = pressures[i];
        float di = densities[i];
        glm::vec4 ri = positions[i];
        glm::vec3 vi = velocities[i];

        glm::vec3 pforce = glm::vec3(0.0f);
        glm::vec3 vforce = glm::vec3(0.0f);
        glm::vec3 pos = glm::vec3(ri);
        glm::ivec3 cellIdx = glm::floor(pos / h);

        forall_fluid_neighbors_unsorted
        (
            glm::vec3 rij = glm::vec3(ri - positions[j]);

            pforce -= (pressures[j] / (densities[j] * densities[j]) + pi / (di * di)) * spikyW(rij, h, spikyConst);
            vforce += (velocities[j] - vi) / densities[j] * laplW(rij, h, spikyConst);
        );

        forces[i] += (vforce * visco + pforce) * mass;
    }
}

__global__ void integration(glm::vec4* positions, glm::vec3* forces, glm::vec3* velocities, glm::vec4* colors, float* timeStep_ptr, int* size_ptr)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int size = *size_ptr;

    if (i < size)
    {
        float timeStep = *timeStep_ptr;
        glm::vec3 G = glm::vec3(0, -9.8f, 0);

        velocities[i] += timeStep * (forces[i] + G);
        positions[i] += glm::vec4(timeStep * velocities[i], 0.0f);

        forces[i] = glm::vec3(0.0f);

        // Color por velocidad
        float speed = glm::length(velocities[i]);
        const float maxSpeed = 4.0f; 
        glm::vec3 hsvMin = glm::vec3(210.0f, 1.0f, 1.0f); 
        glm::vec3 hsvMax = glm::vec3(210.0f, 0.13f, 1.0);
        glm::vec3 hsv = glm::mix(hsvMin, hsvMax, speed / maxSpeed);
        colors[i] = glm::vec4(glm::rgbColor(hsv), 1.0f);
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

__global__ void resetOffset(uint32_t* cellOffset, int* size_ptr)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int size = *size_ptr * 2;

    if (i < size)
    {
        cellOffset[i] = size;
    }
}

__global__ void resetParticleIndexBuffer(uint32_t* particleIndexBuffer, int* size_ptr)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int size = *size_ptr;

    if (i < size)
    {
        particleIndexBuffer[i] = i; 
    }
}

__global__ void insertParticlesUnsorted(glm::vec4* positions, uint32_t* particleIndexBuffer, uint32_t* cellIndexBuffer, glm::vec4* colors, float* h_ptr, int* size_ptr)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int size = *size_ptr;

    if (i < size)
    {
        float h = *h_ptr;
        //uint32_t idx = particleIndexBuffer[i]; // Si en cada iteracion se resetea particleIndexBuffer no hace falta coger el id de la partícula de particleIndexBuffer
        uint32_t idx = i;
        glm::vec3 pos = glm::vec3(positions[idx]);

        glm::ivec3 cell = glm::floor(pos / h);
        uint32_t hashedCell = getHashedCell(cell, size * 2);
        cellIndexBuffer[idx] = hashedCell;
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
        colors[i] = colorById(hashedCell);
    }
}

__global__ void computeCellOffsetUnsorted(uint32_t* particleIndexBuffer, uint32_t* cellIndexBuffer, uint32_t* cellOffsetBuffer, int* size_ptr)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int size = *size_ptr;

    if (i < size)
    {
        uint32_t hashedCell = cellIndexBuffer[particleIndexBuffer[i]];
        atomicMin(&cellOffsetBuffer[hashedCell], uint32_t(i));
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

        volumes[i - fluidSize] = 1.0f / delta; // Volume tiene tamaño boundarySize
    }
}

void SPHSolver::step(VAO_t vao)
{
    stepSorted(vao);
    //stepUnsorted(vao);
}

void SPHSolver::stepSorted(VAO_t vao)
{
    size_t bytes;
    glm::vec4* d_positions;
    glm::vec4* d_colors;

    // Map resources (positions and colors)
    gpuErrchk(cudaGraphicsMapResources(1, &vao.cuda_p_id, 0)); 
    gpuErrchk(cudaGraphicsMapResources(1, &vao.cuda_c_id, 0)); 
    // Get pointers of mapped data (positions and colors)
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&d_positions, &bytes, vao.cuda_p_id)); 
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&d_colors, &bytes, vao.cuda_c_id)); 

    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((unsigned) ceil(2 * *size / (float) BLOCK_SIZE));

    // Reset celloffsets
    resetOffset<<<gridDim, blockDim>>>(d_cellOffsetBuffer, d_size);

    gridDim = dim3((unsigned) ceil(*fluidSize / (float) BLOCK_SIZE));
    
    // Insert Particles 
    insertParticles<<<gridDim, blockDim>>>(d_positions, d_cellIndexBuffer, d_colors, d_h, d_boundarySize, d_fluidSize);

    // Ordenar cellIndexBuffer, positions y velocities
    thrust::device_ptr<uint32_t> cellIndexPtr = thrust::device_pointer_cast(d_cellIndexBuffer);
    thrust::device_ptr<glm::vec4> posPtr = thrust::device_pointer_cast(d_positions);
    thrust::device_ptr<glm::vec3> velPtr = thrust::device_pointer_cast(d_velocities);
    thrust::device_ptr<glm::vec4> colPtr = thrust::device_pointer_cast(d_colors); // Solo es necesario ordenar los colores si se pinta por id o cellid
    thrust::sort_by_key(cellIndexPtr, cellIndexPtr + (*fluidSize), thrust::make_zip_iterator(thrust::make_tuple(posPtr, velPtr, colPtr)));

    // Cell offsets
    computeCellOffset<<<gridDim, blockDim>>>(d_cellIndexBuffer, d_cellOffsetBuffer, d_fluidSize);

    computeDensity<<<gridDim, blockDim>>>(d_positions, d_densities, d_pressures, d_volumes, d_cellIndexBuffer, d_cellOffsetBuffer, d_cellOffsetBoundary, d_h, d_mass, d_density0, d_cubicConstK, d_stiffness, d_boundarySize, d_fluidSize);
    gpuErrchk(cudaGetLastError());

    computeForces<<<gridDim, blockDim>>>(d_positions, d_forces, d_velocities, d_densities, d_pressures, d_volumes, d_cellIndexBuffer, d_cellOffsetBuffer, d_cellOffsetBoundary, d_density0, d_h, d_mass, d_spikyConst, d_viscosity, d_boundarySize, d_fluidSize);
    gpuErrchk(cudaGetLastError());

    integration<<<gridDim, blockDim>>>(d_positions, d_forces, d_velocities, d_colors, d_timeStep, d_fluidSize);
    gpuErrchk(cudaGetLastError());

    //simpleBoundaryCondition<<<gridDim, blockDim>>>(d_positions, d_velocities, d_minDomain, d_maxDomain, d_fluidSize);
    //gpuErrchk(cudaGetLastError());

    // Unmap resources
    gpuErrchk(cudaGraphicsUnmapResources(1, &vao.cuda_p_id, 0)); 
    gpuErrchk(cudaGraphicsUnmapResources(1, &vao.cuda_c_id, 0)); 
}

void SPHSolver::stepUnsorted(VAO_t vao)
{
    size_t bytes;
    glm::vec4* d_positions;
    glm::vec4* d_colors;

    // Map resources (positions and colors)
    gpuErrchk(cudaGraphicsMapResources(1, &vao.cuda_p_id, 0)); 
    gpuErrchk(cudaGraphicsMapResources(1, &vao.cuda_c_id, 0)); 
    // Get pointers of mapped data (positions and colors)
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&d_positions, &bytes, vao.cuda_p_id)); 
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&d_colors, &bytes, vao.cuda_c_id)); 

    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((unsigned) ceil(2 * *size / (float) BLOCK_SIZE));

    // Reset celloffsets
    resetOffset<<<gridDim, blockDim>>>(d_cellOffsetBuffer, d_size);

    gridDim = dim3((unsigned) ceil(*size / (float) BLOCK_SIZE));

    // Reset particle index buffer
    resetParticleIndexBuffer<<<gridDim, blockDim>>>(d_particleIndexBuffer, d_size);
    
    // Insert Particles 
    insertParticlesUnsorted<<<gridDim, blockDim>>>(d_positions, d_particleIndexBuffer, d_cellIndexBuffer, d_colors, d_h, d_size);

    thrust::device_ptr<uint32_t> cellIndexPtr = thrust::device_pointer_cast(d_cellIndexBuffer);
    thrust::device_ptr<uint32_t> particleIndexPtr = thrust::device_pointer_cast(d_particleIndexBuffer);

    // Versión sin ordenación
    compare_cells comp;
    comp.cellIndexBuffer = d_cellIndexBuffer;
    thrust::sort(particleIndexPtr, particleIndexPtr + (*size), comp);

    // Cell offsets
    computeCellOffsetUnsorted<<<gridDim, blockDim>>>(d_particleIndexBuffer, d_cellIndexBuffer, d_cellOffsetBuffer, d_size);

    computeDensityUnsorted<<<gridDim, blockDim>>>(d_positions, d_densities, d_pressures, d_particleIndexBuffer, d_cellIndexBuffer, d_cellOffsetBuffer, d_h, d_mass, d_density0, d_cubicConstK, d_stiffness, d_size);
    gpuErrchk(cudaGetLastError());

    computeForcesUnsorted<<<gridDim, blockDim>>>(d_positions, d_forces, d_velocities, d_densities, d_pressures, d_particleIndexBuffer, d_cellIndexBuffer, d_cellOffsetBuffer, d_h, d_mass, d_spikyConst, d_viscosity, d_size);
    gpuErrchk(cudaGetLastError());

    integration<<<gridDim, blockDim>>>(d_positions, d_forces, d_velocities, d_colors, d_timeStep, d_size);
    gpuErrchk(cudaGetLastError());

    simpleBoundaryCondition<<<gridDim, blockDim>>>(d_positions, d_velocities, d_minDomain, d_maxDomain, d_size);
    gpuErrchk(cudaGetLastError());

    // Unmap resources
    gpuErrchk(cudaGraphicsUnmapResources(1, &vao.cuda_p_id, 0)); 
    gpuErrchk(cudaGraphicsUnmapResources(1, &vao.cuda_c_id, 0)); 
}

void SPHSolver::init()
{
    float h = *this->h;
    int fluidSize = *this->fluidSize;
    int boundarySize = *this->boundarySize; // Si se utiliza la version sin ordenación boundarySize debe ser 0
    int size = *this->size;
    
    // Variables que no existen fuera de la clase SPHSolver y por lo tanto se reserva memoria para ellas
    this->spikyConst = new float(45.0f / ((float) (M_PI * pow(h, 6.0))));
    this->cubicConstK = new float(8.0f / ((float) M_PI * h * h * h));   
    this->particleIndexBuffer = new uint32_t[fluidSize]; // Solo en la version sin ordenacion
    this->cellIndexBuffer = new uint32_t[fluidSize + boundarySize]; // fluid y boundary comparten este array

    // Numero de celdas igual al doble del tamaño total (2 * (fluidSize + boundarySize))
    this->cellOffsetBuffer = new uint32_t[2 * size];
    this->cellOffsetBoundary = new uint32_t[2 * size];

    this->volumes = new float[boundarySize];

    for (int i = 0; i < fluidSize; ++i)
    {
        particleIndexBuffer[i] = (uint32_t) i;
    }  

    for (int i = 0; i < boundarySize; ++i)
    {
        volumes[i] = 0.0f;
    }  

    allocateCudaMemory();
}

void SPHSolver::reset(VAO_t vao, glm::vec4* h_positions)
{
    size_t bytes;
    glm::vec4* d_positions;

    gpuErrchk(cudaGraphicsMapResources(1, &vao.cuda_p_id, 0)); // Map resources
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&d_positions, &bytes, vao.cuda_p_id)); // Get pointer of mapped data

    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((unsigned) ceil(*size / (float) BLOCK_SIZE));
    resetCuda<<<gridDim, blockDim>>>(d_velocities, d_forces, d_size);
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaGraphicsUnmapResources(1, &vao.cuda_p_id, 0)); // Unmap resources

    // Copy positions
    gpuErrchk(cudaMemcpy(d_positions, h_positions, sizeof(glm::vec4) * *size, cudaMemcpyHostToDevice));

    precomputeBoundaryNeighbors(vao);
}

void SPHSolver::precomputeBoundaryNeighbors(VAO_t vao)
{
    size_t bytes;
    glm::vec4* d_positions;
    glm::vec4* d_colors;

    // Map resources (positions and colors)
    gpuErrchk(cudaGraphicsMapResources(1, &vao.cuda_p_id, 0)); 
    gpuErrchk(cudaGraphicsMapResources(1, &vao.cuda_c_id, 0)); 
    // Get pointers of mapped data (positions and colors)
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&d_positions, &bytes, vao.cuda_p_id)); 
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&d_colors, &bytes, vao.cuda_c_id)); 

    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((unsigned) ceil(2 * *size / (float) BLOCK_SIZE));

    // Reset boundary cell offsets
    resetOffset<<<gridDim, blockDim>>>(d_cellOffsetBoundary, d_size);
    gpuErrchk(cudaGetLastError());

    gridDim = dim3((unsigned) ceil(*boundarySize / (float) BLOCK_SIZE));
    
    // Insert boundary particles 
    insertBoundaryParticles<<<gridDim, blockDim>>>(d_positions, d_cellIndexBuffer, d_colors, d_h, d_boundarySize, d_fluidSize);
    gpuErrchk(cudaGetLastError());

    // Ordenar cellIndexBuffer, positions y velocities (La parte de los arrays correspondiente a las boundary particles [fluidSize, size])
    thrust::device_ptr<uint32_t> cellIndexPtr = thrust::device_pointer_cast(d_cellIndexBuffer + *fluidSize);
    thrust::device_ptr<glm::vec4> posPtr = thrust::device_pointer_cast(d_positions + *fluidSize);
    thrust::device_ptr<glm::vec3> velPtr = thrust::device_pointer_cast(d_velocities + *fluidSize);
    thrust::device_ptr<glm::vec4> colPtr = thrust::device_pointer_cast(d_colors + *fluidSize);
    thrust::sort_by_key(cellIndexPtr, cellIndexPtr + (*boundarySize), thrust::make_zip_iterator(thrust::make_tuple(posPtr, velPtr, colPtr)));

    // Cell offsets
    computeBoundaryCellOffset<<<gridDim, blockDim>>>(d_cellIndexBuffer, d_cellOffsetBoundary, d_boundarySize, d_fluidSize);
    gpuErrchk(cudaGetLastError());

    // Compute volume
    computeVolume<<<gridDim, blockDim>>>(d_positions, d_volumes, d_cellIndexBuffer, d_cellOffsetBoundary, d_h, d_cubicConstK, d_boundarySize, d_fluidSize);
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaGraphicsUnmapResources(1, &vao.cuda_p_id, 0)); // Unmap resources
    gpuErrchk(cudaGraphicsUnmapResources(1, &vao.cuda_c_id, 0)); 
}

void SPHSolver::allocateCudaMemory()
{
    // Reservar memoria en GPU
    gpuErrchk(cudaMalloc(&d_h, sizeof(float)));
    gpuErrchk(cudaMalloc(&d_timeStep, sizeof(float)));
    gpuErrchk(cudaMalloc(&d_cubicConstK, sizeof(float)));
    gpuErrchk(cudaMalloc(&d_spikyConst, sizeof(float)));
    gpuErrchk(cudaMalloc(&d_radius, sizeof(float)));
    gpuErrchk(cudaMalloc(&d_mass, sizeof(float)));
    gpuErrchk(cudaMalloc(&d_density0, sizeof(float)));
    gpuErrchk(cudaMalloc(&d_stiffness, sizeof(float)));
    gpuErrchk(cudaMalloc(&d_viscosity, sizeof(float)));
    gpuErrchk(cudaMalloc(&d_size, sizeof(int)));
    gpuErrchk(cudaMalloc(&d_fluidSize, sizeof(int)));
    gpuErrchk(cudaMalloc(&d_boundarySize, sizeof(int)));
    gpuErrchk(cudaMalloc(&d_densities, sizeof(float) * *size));
    gpuErrchk(cudaMalloc(&d_pressures, sizeof(float) * *size));
    gpuErrchk(cudaMalloc(&d_forces, sizeof(glm::vec3) * *size));
    gpuErrchk(cudaMalloc(&d_velocities, sizeof(glm::vec3) * *size));
    gpuErrchk(cudaMalloc(&d_minDomain, sizeof(glm::vec3)));
    gpuErrchk(cudaMalloc(&d_maxDomain, sizeof(glm::vec3)));
    gpuErrchk(cudaMalloc(&d_cellIndexBuffer, sizeof(uint32_t) * (*size)));
    gpuErrchk(cudaMalloc(&d_particleIndexBuffer, sizeof(uint32_t) * (*fluidSize)));
    gpuErrchk(cudaMalloc(&d_cellOffsetBuffer, sizeof(uint32_t) * (*size * 2)));
    gpuErrchk(cudaMalloc(&d_cellOffsetBoundary, sizeof(uint32_t) * (*size * 2)));
    gpuErrchk(cudaMalloc(&d_volumes, sizeof(float) * (*boundarySize)));

    // Copiar datos a GPU
    gpuErrchk(cudaMemcpy(d_h, h, sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_timeStep, timeStep, sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_cubicConstK, cubicConstK, sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_spikyConst, spikyConst, sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_radius, radius, sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_mass, mass, sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_density0, density0, sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_stiffness, stiffness, sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_viscosity, viscosity, sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_size, size, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_fluidSize, fluidSize, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_boundarySize, boundarySize, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_densities, densities, sizeof(float) * (*size), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_pressures, pressures, sizeof(float) * (*size), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_forces, forces, sizeof(float) * (*size), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_velocities, velocities, sizeof(float) * (*size), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_minDomain, minDomain, sizeof(glm::vec3), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_maxDomain, maxDomain, sizeof(glm::vec3), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_cellIndexBuffer, cellIndexBuffer, sizeof(uint32_t) * (*size), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particleIndexBuffer, particleIndexBuffer, sizeof(uint32_t) * (*fluidSize), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_cellOffsetBuffer, cellOffsetBuffer, sizeof(uint32_t) * (*size * 2), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_cellOffsetBoundary, cellOffsetBoundary, sizeof(uint32_t) * (*size * 2), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_volumes, volumes, sizeof(float) * (*boundarySize), cudaMemcpyHostToDevice));

    // Copy constant symbol
    glm::ivec3 displacements[] = { glm::ivec3(-1, -1, -1), glm::ivec3(-1, -1, 0), glm::ivec3(-1, -1, 1), glm::ivec3(-1, 0, -1), glm::ivec3(-1, 0, 0), glm::ivec3(-1, 0, 1), glm::ivec3(-1, 1, -1), glm::ivec3(-1, 1, 0), glm::ivec3(-1, 1, 1), glm::ivec3(0, -1, -1), glm::ivec3(0, -1, 0), glm::ivec3(0, -1, 1), glm::ivec3(0, 0, -1), glm::ivec3(0, 0, 0), glm::ivec3(0, 0, 1), glm::ivec3(0, 1, -1), glm::ivec3(0, 1, 0), glm::ivec3(0, 1, 1), glm::ivec3(1, -1, -1), glm::ivec3(1, -1, 0), glm::ivec3(1, -1, 1), glm::ivec3(1, 0, -1), glm::ivec3(1, 0, 0), glm::ivec3(1, 0, 1), glm::ivec3(1, 1, -1), glm::ivec3(1, 1, 0), glm::ivec3(1, 1, 1)};
    cudaMemcpyToSymbol(NEIGH_DISPLACEMENTS, &displacements[0], sizeof(glm::ivec3) * 27);
}

void SPHSolver::freeCudaMemory()
{
    gpuErrchk(cudaFree(d_h));
    gpuErrchk(cudaFree(d_cubicConstK));
    gpuErrchk(cudaFree(d_spikyConst));
    gpuErrchk(cudaFree(d_radius));
    gpuErrchk(cudaFree(d_mass));
    gpuErrchk(cudaFree(d_stiffness));
    gpuErrchk(cudaFree(d_viscosity));
    gpuErrchk(cudaFree(d_size));
    gpuErrchk(cudaFree(d_densities));
    gpuErrchk(cudaFree(d_pressures));
    gpuErrchk(cudaFree(d_forces));
    gpuErrchk(cudaFree(d_velocities));
    gpuErrchk(cudaFree(d_minDomain));
    gpuErrchk(cudaFree(d_maxDomain));
    gpuErrchk(cudaFree(d_cellIndexBuffer));
    gpuErrchk(cudaFree(d_particleIndexBuffer));
    gpuErrchk(cudaFree(d_cellOffsetBuffer));
}

void SPHSolver::release()
{
    freeCudaMemory();

    delete spikyConst;
    delete cubicConstK;
    delete[] particleIndexBuffer;
    delete[] cellIndexBuffer;
    delete[] cellOffsetBuffer;
}



