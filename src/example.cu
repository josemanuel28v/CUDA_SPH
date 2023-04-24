#include "example.cuh"
#include "stdio.h"
#include "types.h"

__global__ void particles(glm::vec4* positions, int numParticles)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < numParticles)
    {
        positions[idx].y += 0.01f;
    }
}

void moveParticles(glm::vec4* positions, int numParticles)
{
    uint threads_per_block = 16;
    dim3 blockDim(threads_per_block);
    dim3 gridDim(ceil(numParticles / (float) threads_per_block));
    particles<<<gridDim, blockDim>>>(positions, numParticles);
}


__global__ void sumArray(float *a, float* b, float* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Cuando solo hay una dimension 

    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void sumReduction(float *arr, float *sum, int size) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float local_sum = 0.0f;
    for (int i = tid; i < size; i += stride) 
    {
        local_sum += arr[i];
    }
    
    atomicAdd(sum, local_sum);
}

// __global__ void sumReduction(float* a, float *sum, int n)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x; // Cuando solo hay una dimension 

//     if (idx < n)
//     {
//         *sum += a[idx];
//         atomicAdd(sum, a[idx]);
//     }
// }

#define THREADS_PER_BLOCK 8

void example1()
{
    const int N = 100;
    const int NUM_BYTES = N * sizeof(float);

    // Data pointers
    float* h_a;
    float* h_b;
    float* h_c;
    float* d_a;
    float* d_b;
    float* d_c;

    // Allocate host memory
    h_a = (float*)malloc(NUM_BYTES);
    h_b = (float*)malloc(NUM_BYTES);
    h_c = (float*)malloc(NUM_BYTES);

    // Allocate device memory
    cudaMalloc((void**)&d_a, NUM_BYTES);
    cudaMalloc((void**)&d_b, NUM_BYTES);
    cudaMalloc((void**)&d_c, NUM_BYTES);

    // Data initialization
    for (int i = 0; i < N; ++i) h_a[i] = 1.0f;
    for (int i = 0; i < N; ++i) h_b[i] = 2.0f;

    cudaMemcpy(d_a, h_a, NUM_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, NUM_BYTES, cudaMemcpyHostToDevice);

    // Grid
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim((uint) ceil(N / (float) THREADS_PER_BLOCK));

    // Run kernel
    sumArray<<<gridDim, blockDim>>>(d_a, d_b, d_c, N);

    // Copy result from GPU to CPU 
    cudaMemcpy(h_c, d_c, NUM_BYTES, cudaMemcpyDeviceToHost);

    // Show result
    for (int i = 0; i < N; ++i)
    {
        printf("Pos %d %f \n", i, h_c[i]);
    }

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
}

void example2()
{
    const int N = 100;
    const int NUM_BYTES = N * sizeof(float);

    // Data pointers
    float* h_a;
    float* h_sum;
    float* d_a;
    float* d_sum;

    // Allocate host memory
    h_a = (float*)malloc(NUM_BYTES);
    h_sum = (float*)malloc(sizeof(float));

    // Allocate device memory
    cudaMalloc((void**)&d_a, NUM_BYTES);
    cudaMalloc((void**)&d_sum, sizeof(float));

    // Data initialization
    for (int i = 0; i < N; ++i) h_a[i] = 1.0f;
    *h_sum = 0.0f;

    cudaMemcpy(d_a, h_a, NUM_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, h_sum, sizeof(float), cudaMemcpyHostToDevice);

    // Grid
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim((uint) ceil(N / (float) THREADS_PER_BLOCK));

    // Run kernel
    sumReduction<<<gridDim, blockDim>>>(d_a, d_sum, N);

    // Copy result from GPU to CPU 
    cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    // Check result
    if (*h_sum == N) 
    {   

        printf("Correcto! \n");
    }
    else
    { 
        printf(" Incorrecto: %f != %d \n", *h_sum, N);
    }

    // Free memory
    cudaFree(d_a);
    cudaFree(d_sum);
    free(h_a);
    free(h_sum);
}