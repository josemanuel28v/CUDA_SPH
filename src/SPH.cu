#include "SPH.cuh"
#include "SPHKernels.cuh"
#include "stdio.h"
#include "types.h"

#define BLOCK_SIZE 128

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

        float h2 = h * h; // Precalcular
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

__global__ void integrationCuda(glm::vec4* positions, glm::vec3* forces, glm::vec3* velocities, float* timeStep_ptr, int* size_ptr)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int size = *size_ptr;

    if (i < size)
    {
        float timeStep = *timeStep_ptr; // fixed step
        glm::vec3 G = glm::vec3(0, -9.8f, 0);

        velocities[i] += timeStep * (forces[i] + G);
        positions[i] += glm::vec4(timeStep * velocities[i], 0.0f);

        forces[i] = glm::vec3(0.0f);
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

void SPH::checkValues()
{
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(ceil(*size / (float) BLOCK_SIZE));
    checkValuesCuda<<<gridDim, blockDim>>>(d_h, d_mass);
    gpuErrchk(cudaGetLastError());
}

void SPH::step(cudaGraphicsResource* positionBufferObject)
{
    size_t bytes;
    glm::vec4* d_positions;

    gpuErrchk(cudaGraphicsMapResources(1, &positionBufferObject, 0)); // Map resources
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&d_positions, &bytes, positionBufferObject)); // Get pointer of mapped data

    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(ceil(*size / (float) BLOCK_SIZE));

    computeDensityCuda<<<gridDim, blockDim>>>(d_positions, d_densities, d_pressures, d_h, d_mass, d_density0, d_cubicConstK, d_stiffness, d_size);
    gpuErrchk(cudaGetLastError());

    computePressureForceCuda<<<gridDim, blockDim>>>(d_positions, d_forces, d_velocities, d_densities, d_pressures, d_h, d_mass, d_density0, d_spikyConst, d_viscosity, d_size);
    gpuErrchk(cudaGetLastError());

    //computeViscosityForceCuda<<<gridDim, blockDim>>>(d_positions, d_forces, d_velocities, d_densities, d_h, d_mass, d_viscosity, d_spikyConst, d_size);
    //gpuErrchk(cudaGetLastError());

    integrationCuda<<<gridDim, blockDim>>>(d_positions, d_forces, d_velocities, d_timeStep, d_size);
    gpuErrchk(cudaGetLastError());

    simpleBoundaryConditionCuda<<<gridDim, blockDim>>>(d_positions, d_velocities, d_minDomain, d_maxDomain, d_size);
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaGraphicsUnmapResources(1, &positionBufferObject, 0)); // Unmap resources
}

void SPH::init()
{
    float h = *this->h;
    // Variables que no existen fuera de la clase SPH y por lo tanto se reserva memoria para ellas
    //this->poly6Const = new float(315.0f / (64.0f * M_PI * pow(h, 9))); // delete in destructor
    this->spikyConst = new float(45.0f / (M_PI * pow(h, 6))); // delete in destructor
    this->cubicConstK = new float(8.0f / (M_PI * h * h * h));   

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
}

