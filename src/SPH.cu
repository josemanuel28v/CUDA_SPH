#include "SPH.cuh"
#include "SPHKernels.cuh"
#include "stdio.h"
#include "types.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <climits>
#include <glm/gtx/color_space.hpp>
#include <curand.h>
#include <curand_kernel.h>

__constant__ glm::ivec3 NEIGH_DISPLACEMENTS[27];
#define BLOCK_SIZE 128
#define forall_fluid_neighbors(code)\
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

// Funciona más rápido con el operador xor que con el operador + pero con xor ocurren cosas raras a las particulas
// (con 80000 particulas y operador + 50 fps aprox)
// __device__ uint32_t getHashedCell(glm::ivec3 cellIdx, uint32_t size)
// {
//     const uint32_t p1 = 73856093; // some large primes
//     const uint32_t p2 = 19349663;
//     const uint32_t p3 = 83492791;

//     int n = p1 * cellIdx.x ^ p2 *cellIdx.y ^ p3 * cellIdx.z;
//     n %= size;

//     return n;
// }

// Más rápido que el anterior (con 80000 particulas 71 fps aprox)
__device__ uint32_t getHashedCell1(const glm::ivec3 key, const uint32_t size) 
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

// Admite negativos (con 80000 particulas 79 fps aprox)
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

__global__ void computeDensityCuda(glm::vec4* positions, float* densities, float* pressures, float* h_ptr, float* mass_ptr, float* density0_ptr, float* cubicConst_ptr, float* stiffness_ptr, int* size_ptr)
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

        float density = 0.0f;

        for (int j = 0; j < size; ++j) // iterar sobre los "vecinos"
        {
            glm::vec3 rij = glm::vec3(ri - positions[j]);
            density += mass * cubicW(rij, h, cubicConst);
        }

        densities[i] = density;
        pressures[i] = max(0.0f, stiffness * (density - density0));
    }
}

__global__ void computeDensityHashed(glm::vec4* positions, float* densities, float* pressures, uint32_t* particleIndexBuffer, uint32_t* cellIndexBuffer, uint32_t* cellOffsetBuffer, float* h_ptr, float* mass_ptr, float* density0_ptr, float* cubicConst_ptr, float* stiffness_ptr, int* size_ptr)
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
        glm::vec3 pos = glm::vec3(positions[i]);

        float density = 0.0f;

        glm::ivec3 cellIdx = glm::floor(pos / h);

        forall_fluid_neighbors
        (
            glm::vec3 rij = glm::vec3(ri - positions[j]);
            density += mass * cubicW(rij, h, cubicConst);
        );

        densities[i] = density;
        pressures[i] = max(0.0f, stiffness * (density - density0));
    }
}

__global__ void computePressureForceCuda(glm::vec4* positions, glm::vec3* forces, glm::vec3* velocities, float* densities, float* pressures, float* h_ptr, float* mass_ptr, float* density0_ptr, float* spikyConst_ptr, float* viscosity_ptr, int* size_ptr)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int size = *size_ptr;

    if (i < size)
    {
        float h = *h_ptr;
        float mass = *mass_ptr;
        float density0 = *density0_ptr;
        float spikyConst = *spikyConst_ptr;
        float visco = *viscosity_ptr;
        float pi = pressures[i];
        float di = densities[i];
        glm::vec4 ri = positions[i];
        glm::vec3 vi = velocities[i];

        glm::vec3 pforce = glm::vec3(0.0f);
        glm::vec3 vforce = glm::vec3(0.0f);

        for (int j = 0; j < size; ++j)
        {
            glm::vec3 rij = glm::vec3(ri - positions[j]);
            float pj = pressures[j];
            float dj = densities[j];
            pforce -= (pj / (dj * dj) + pi / (di * di)) * spikyW(rij, h, spikyConst);
            vforce += (velocities[j] - vi) / densities[j] * laplW(rij, h, spikyConst);
        }

        forces[i] += pforce * mass;
        forces[i] += vforce * visco * mass;
    }
}

__global__ void computePressureForceCudaHashed(glm::vec4* positions, glm::vec3* forces, glm::vec3* velocities, float* densities, float* pressures, uint32_t* particleIndexBuffer, uint32_t* cellIndexBuffer, uint32_t* cellOffsetBuffer, float* h_ptr, float* mass_ptr, float* density0_ptr, float* spikyConst_ptr, float* viscosity_ptr, int* size_ptr)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int size = *size_ptr;

    if (i < size)
    {
        float h = *h_ptr;
        float mass = *mass_ptr;
        float density0 = *density0_ptr;
        float spikyConst = *spikyConst_ptr;
        float visco = *viscosity_ptr;
        float pi = pressures[i];
        float di = densities[i];
        glm::vec4 ri = positions[i];
        glm::vec3 vi = velocities[i];

        glm::vec3 pforce = glm::vec3(0.0f);
        glm::vec3 vforce = glm::vec3(0.0f);
        glm::vec3 pos = glm::vec3(positions[i]);
        glm::ivec3 cellIdx = glm::floor(pos / h);

        forall_fluid_neighbors
        (
            glm::vec3 rij = glm::vec3(ri - positions[j]);
            float pj = pressures[j];
            float dj = densities[j];

            pforce -= (pj / (dj * dj) + pi / (di * di)) * spikyW(rij, h, spikyConst);
            vforce += (velocities[j] - vi) / dj * laplW(rij, h, spikyConst);
        );

        forces[i] += pforce * mass;
        forces[i] += vforce * visco * mass;
    }
}

__global__ void computeViscosityForceCuda(glm::vec4* positions, glm::vec3* forces, glm::vec3* velocities, float* densities, float* h_ptr, float* mass_ptr, float* visco_ptr, float* laplConst_ptr, int* size_ptr)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int size = *size_ptr;

    if (i < size)
    {
        float h = *h_ptr;
        float mass = *mass_ptr;
        float visco = *visco_ptr;
        float laplConst = *laplConst_ptr;
        glm::vec4 ri = positions[i];
        glm::vec3 vi = velocities[i];

        glm::vec3 force = glm::vec3(0.0f);

        for (int j = 0; j < size; ++j)
        {
            glm::vec3 rij = glm::vec3(ri - positions[j]);
            force += mass * (velocities[j] - vi) / densities[j] * laplW(rij, h, laplConst);
        }

        forces[i] += force * visco;
    }
}

__global__ void integrationCuda(glm::vec4* positions, glm::vec3* forces, glm::vec3* velocities, glm::vec4* colors, float* timeStep_ptr, int* size_ptr)
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

        // Coloreado con velocidad
        float speed = glm::length(velocities[i]);
        const float maxSpeed = 4.0f; 
        glm::vec3 hsvMin = glm::vec3(210.0f, 1.0f, 1.0f); // azul
        glm::vec3 hsvMax = glm::vec3(210.0f, 0.13f, 1.0); // azul casi blanco
        glm::vec3 hsv = glm::mix(hsvMin, hsvMax, speed / maxSpeed);
        colors[i] = glm::vec4(glm::rgbColor(hsv), 1.0f);
    }
}

__global__ void simpleBoundaryConditionCuda(glm::vec4* positions, glm::vec3* velocities, glm::vec3* min_ptr, glm::vec3* max_ptr, int* size_ptr)
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

__global__ void checkValuesCuda(float* h_ptr, float* mass_ptr)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i == 0)
    {
        printf("h %f\n", *h_ptr);
        printf("mass %f\n", *mass_ptr);
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

__global__ void resetparticleIndexBuffer(uint32_t* particleIndexBuffer, int* size_ptr)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int size = *size_ptr;

    if (i < size)
    {
        particleIndexBuffer[i] = i; 
    }
}

__global__ void insertParticles(glm::vec4* positions, uint32_t* particleIndexBuffer, uint32_t* cellIndexBuffer, glm::vec4* colors, float* h_ptr, int* size_ptr)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int size = *size_ptr;

    if (i < size)
    {
        float h = *h_ptr;
        uint32_t idx = particleIndexBuffer[i];
        glm::vec3 pos = glm::vec3(positions[idx]);

        glm::ivec3 cell = glm::floor(pos / h);
        uint32_t hashedCell = getHashedCell(cell, size * 2);
        cellIndexBuffer[idx] = hashedCell;

        // Color por celda
        // curandState_t state;
        // curand_init(blockIdx.x, 0, 0, &state); // inicializar la semilla con el id de partícula
        // glm::vec4 randColor = glm::vec4(
        //     curand_uniform(&state),
        //     curand_uniform(&state),
        //     curand_uniform(&state),
        //     1.0f
        // );
        // colors[idx] = randColor;

    }
}

__global__ void computeCellOffset(uint32_t* particleIndexBuffer, uint32_t* cellIndexBuffer, uint32_t* cellOffsetBuffer, int* size_ptr)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int size = *size_ptr;

    if (i < size)
    {
        uint32_t hashedCell = cellIndexBuffer[particleIndexBuffer[i]];
        atomicMin(&cellOffsetBuffer[hashedCell], uint32_t(i));
    }
}

void SPH::checkValues()
{
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(ceil(*size / (float) BLOCK_SIZE));
    checkValuesCuda<<<gridDim, blockDim>>>(d_h, d_mass);
    gpuErrchk(cudaGetLastError());
}

void SPH::step(VAO_t vao)
{
    size_t bytes;
    glm::vec4* d_positions;
    glm::vec4* d_colors;

    gpuErrchk(cudaGraphicsMapResources(1, &vao.cuda_p_id, 0)); // Map resources
    gpuErrchk(cudaGraphicsMapResources(1, &vao.cuda_c_id, 0)); // Map resources
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&d_positions, &bytes, vao.cuda_p_id)); // Get pointer of mapped data
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&d_colors, &bytes, vao.cuda_c_id)); // Get pointer of mapped data

    ////////// NEIGHBOR SEARCH ///////////
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(ceil(2 * *size / (float) BLOCK_SIZE));

    // Reset celloffsets
    resetOffset<<<gridDim, blockDim>>>(d_cellOffsetBuffer, d_size);

    blockDim = dim3(BLOCK_SIZE);
    gridDim = dim3(ceil(*size / (float) BLOCK_SIZE));

    // Reset particle index buffer
    resetparticleIndexBuffer<<<gridDim, blockDim>>>(d_particleIndexBuffer, d_size);
    
    // Insert Particles (Cuidado ivec3 cell negativo)
    insertParticles<<<gridDim, blockDim>>>(d_positions, d_particleIndexBuffer, d_cellIndexBuffer, d_colors, d_h, d_size);

    // Sort
    thrust::device_ptr<uint32_t> particleIndexPtr = thrust::device_pointer_cast(d_particleIndexBuffer);
    thrust::device_ptr<uint32_t> cellIndexPtr = thrust::device_pointer_cast(d_cellIndexBuffer);
    thrust::sort(particleIndexPtr, particleIndexPtr + *size, [cellIndexPtr] __device__ (int a, int b) { 
        return cellIndexPtr[a] < cellIndexPtr[b]; 
    });

    // Cell offsets
    computeCellOffset<<<gridDim, blockDim>>>(d_particleIndexBuffer, d_cellIndexBuffer, d_cellOffsetBuffer, d_size);

    //computeDensityCuda<<<gridDim, blockDim>>>(d_positions, d_densities, d_pressures, d_h, d_mass, d_density0, d_cubicConstK, d_stiffness, d_size);
    computeDensityHashed<<<gridDim, blockDim>>>(d_positions, d_densities, d_pressures, d_particleIndexBuffer, d_cellIndexBuffer, d_cellOffsetBuffer, d_h, d_mass, d_density0, d_cubicConstK, d_stiffness, d_size);
    gpuErrchk(cudaGetLastError());

    //computePressureForceCuda<<<gridDim, blockDim>>>(d_positions, d_forces, d_velocities, d_densities, d_pressures, d_h, d_mass, d_density0, d_spikyConst, d_viscosity, d_size);
    computePressureForceCudaHashed<<<gridDim, blockDim>>>(d_positions, d_forces, d_velocities, d_densities, d_pressures, d_particleIndexBuffer, d_cellIndexBuffer, d_cellOffsetBuffer, d_h, d_mass, d_density0, d_spikyConst, d_viscosity, d_size);
    gpuErrchk(cudaGetLastError());

    //computeViscosityForceCuda<<<gridDim, blockDim>>>(d_positions, d_forces, d_velocities, d_densities, d_h, d_mass, d_viscosity, d_spikyConst, d_size);
    //gpuErrchk(cudaGetLastError());

    integrationCuda<<<gridDim, blockDim>>>(d_positions, d_forces, d_velocities, d_colors, d_timeStep, d_size);
    gpuErrchk(cudaGetLastError());

    simpleBoundaryConditionCuda<<<gridDim, blockDim>>>(d_positions, d_velocities, d_minDomain, d_maxDomain, d_size);
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaGraphicsUnmapResources(1, &vao.cuda_p_id, 0)); // Unmap resources
    gpuErrchk(cudaGraphicsUnmapResources(1, &vao.cuda_c_id, 0)); // Unmap resources
}

void SPH::init()
{
    float h = *this->h;
    float size = *this->size;

    // Variables que no existen fuera de la clase SPH y por lo tanto se reserva memoria para ellas
    //this->poly6Const = new float(315.0f / (64.0f * M_PI * pow(h, 9))); // delete in destructor
    this->spikyConst = new float(45.0f / (M_PI * pow(h, 6))); // delete in destructor
    this->cubicConstK = new float(8.0f / (M_PI * h * h * h));   
    this->particleIndexBuffer = new uint32_t[size];
    this->cellIndexBuffer = new uint32_t[size];
    this->cellOffsetBuffer = new uint32_t[2 * size];

    for (uint32_t i = 0; i < size; ++i)
    {
        particleIndexBuffer[i] = i;
    }

    allocateCudaMemory();
}

void SPH::reset(cudaGraphicsResource* positionBufferObject, glm::vec4* h_positions)
{
    size_t bytes;
    glm::vec4* d_positions;

    gpuErrchk(cudaGraphicsMapResources(1, &positionBufferObject, 0)); // Map resources
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&d_positions, &bytes, positionBufferObject)); // Get pointer of mapped data

    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(ceil(*size / (float) BLOCK_SIZE));
    resetCuda<<<gridDim, blockDim>>>(d_velocities, d_forces, d_size);
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaGraphicsUnmapResources(1, &positionBufferObject, 0)); // Unmap resources

    // Copy positions
    gpuErrchk(cudaMemcpy(d_positions, h_positions, sizeof(glm::vec4) * *size, cudaMemcpyHostToDevice));
}

void SPH::allocateCudaMemory()
{
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
    gpuErrchk(cudaMalloc(&d_densities, sizeof(float) * *size));
    gpuErrchk(cudaMalloc(&d_pressures, sizeof(float) * *size));
    gpuErrchk(cudaMalloc(&d_forces, sizeof(glm::vec3) * *size));
    gpuErrchk(cudaMalloc(&d_velocities, sizeof(glm::vec3) * *size));
    gpuErrchk(cudaMalloc(&d_minDomain, sizeof(glm::vec3)));
    gpuErrchk(cudaMalloc(&d_maxDomain, sizeof(glm::vec3)));

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
    gpuErrchk(cudaMemcpy(d_densities, densities, sizeof(float) * (*size), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_pressures, pressures, sizeof(float) * (*size), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_forces, forces, sizeof(float) * (*size), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_velocities, velocities, sizeof(float) * (*size), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_minDomain, minDomain, sizeof(glm::vec3), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_maxDomain, maxDomain, sizeof(glm::vec3), cudaMemcpyHostToDevice));

    // grid stuff
    gpuErrchk(cudaMalloc(&d_cellIndexBuffer, sizeof(uint32_t) * (*size)));
    gpuErrchk(cudaMalloc(&d_particleIndexBuffer, sizeof(uint32_t) * (*size)));
    gpuErrchk(cudaMalloc(&d_cellOffsetBuffer, sizeof(uint32_t) * (*size * 2)));

    // es necesario realmente tener estos arrays en el host?
    gpuErrchk(cudaMemcpy(d_cellIndexBuffer, cellIndexBuffer, sizeof(uint32_t) * (*size), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_particleIndexBuffer, particleIndexBuffer, sizeof(uint32_t) * (*size), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_cellOffsetBuffer, cellOffsetBuffer, sizeof(uint32_t) * (*size * 2), cudaMemcpyHostToDevice));

    // Constant symbol
    glm::ivec3 displacements[] = { glm::ivec3(-1, -1, -1), glm::ivec3(-1, -1, 0), glm::ivec3(-1, -1, 1), glm::ivec3(-1, 0, -1), glm::ivec3(-1, 0, 0), glm::ivec3(-1, 0, 1), glm::ivec3(-1, 1, -1), glm::ivec3(-1, 1, 0), glm::ivec3(-1, 1, 1), glm::ivec3(0, -1, -1), glm::ivec3(0, -1, 0), glm::ivec3(0, -1, 1), glm::ivec3(0, 0, -1), glm::ivec3(0, 0, 0), glm::ivec3(0, 0, 1), glm::ivec3(0, 1, -1), glm::ivec3(0, 1, 0), glm::ivec3(0, 1, 1), glm::ivec3(1, -1, -1), glm::ivec3(1, -1, 0), glm::ivec3(1, -1, 1), glm::ivec3(1, 0, -1), glm::ivec3(1, 0, 0), glm::ivec3(1, 0, 1), glm::ivec3(1, 1, -1), glm::ivec3(1, 1, 0), glm::ivec3(1, 1, 1)};
    cudaMemcpyToSymbol(NEIGH_DISPLACEMENTS, &displacements[0], sizeof(glm::ivec3) * 27);
}

void SPH::freeCudaMemory()
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

