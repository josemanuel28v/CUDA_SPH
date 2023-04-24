#include "SPH.cuh"
#include "stdio.h"
#include "types.h"

#define BLOCK_SIZE 16

__global__ void computeDensityCuda(glm::vec4* positions, float* densities, float* h_ptr, float* mass_ptr, float* poly6Const_ptr, int* size_ptr)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int size = *size_ptr;

    if (i < size)
    {
        float h = *h_ptr;
        float mass = *mass_ptr;
        float poly6Const = *poly6Const_ptr;
        float h2 = h * h; // guardar como constante
        float density0 = 1000.0f; // guardar como constante
        float density = 0.0f;

        for (int j = 0; j < size; ++j) // iterar sobre los "vecinos"
        {
            glm::vec3 rij = positions[i] - positions[j];
            float r2 = glm::dot(rij, rij);

            if (r2 < h2)
            {
                density += poly6Const * (h2 - r2) * (h2 - r2) * (h2 - r2);
            }
        }

        densities[i] = max(density0, density * mass);
    }
}


void SPH::computeDensity(glm::vec4* positions)
{
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(ceil(*size / (float) BLOCK_SIZE));
    computeDensityCuda<<<gridDim, blockDim>>>(positions, d_densities, d_h, d_mass, d_poly6Const, d_size);
    gpuErrchk(cudaGetLastError());
}

__global__ void particles(glm::vec4* positions, int* size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < *size)
    {
        positions[idx].y += 0.01f;
    }
}

void SPH::moveParticles(glm::vec4* positions)
{
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(ceil(*size / (float) BLOCK_SIZE));
    particles<<<gridDim, blockDim>>>(positions, d_size);
    gpuErrchk(cudaGetLastError());
}

void SPH::registerHostPointers(float* h, float *radius, float* mass, int* size, float* densities)
{
    this->h = h;
    this->radius = radius;
    this->mass = mass;
    this->densities = densities;
    this->size = size;

    // Variables que no existen fuera de la clase SPH y por lo tanto se reserva memoria para ellas
    this->poly6Const = new float(315.0f / (64.0f * M_PI * pow(*h, 9))); // delete in destructor
}
void SPH::allocateCudaMemory()
{
    gpuErrchk(cudaMalloc(&d_h, sizeof(float)));
    gpuErrchk(cudaMalloc(&d_poly6Const, sizeof(float)));
    gpuErrchk(cudaMalloc(&d_radius, sizeof(float)));
    gpuErrchk(cudaMalloc(&d_mass, sizeof(float)));
    gpuErrchk(cudaMalloc(&d_size, sizeof(int)));
    gpuErrchk(cudaMalloc(&d_densities, sizeof(float) * *size));

    gpuErrchk(cudaMemcpy(d_h, h, sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_poly6Const, poly6Const, sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_radius, radius, sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_mass, mass, sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_size, size, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_densities, densities, sizeof(float) * (*size), cudaMemcpyHostToDevice));

    // demás variables y arrays
}

void SPH::freeCudaMemory()
{
    gpuErrchk(cudaFree(d_h));
    gpuErrchk(cudaFree(d_poly6Const));
    gpuErrchk(cudaFree(d_radius));
    gpuErrchk(cudaFree(d_mass));
    gpuErrchk(cudaFree(d_size));
    gpuErrchk(cudaFree(d_densities));
}

