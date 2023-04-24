#include "SPHSystem.h"
#include "types.h"
#include "VAO_t.h"
#include "System.h"
#include "cuda_gl_interop.h"

SPHSystem::SPHSystem(glm::vec3 min, glm::vec3 max, Particle* prototype)
{
    // hacer bien llamando al constructor padre
    this->prototype = prototype;
    this->size = initFluidBlock(min, max);
    this->solver = new SPH();
    this->densities.resize(this->size);

    // Set host and devices pointers 
    this->solver->registerHostPointers(
        &h, 
        prototype->getRadiusAddress(), 
        &mass, 
        (int*)(&this->size), 
        densities.data()
    );
    this->solver->allocateCudaMemory();
}

uint SPHSystem::initFluidBlock(glm::vec3 min, glm::vec3 max)
{
    float dist = 2.0f * prototype->getRadius();

    glm::ivec3 ppedge = floor((max - min) / dist + 1e-5f);
    uint numParticles = ppedge.x * ppedge.y * ppedge.z;
    
    positions.resize(numParticles);
    uint id = 0;
    for (int i = 0; i < ppedge.x; ++i)
    {
        for (int j = 0; j < ppedge.y; ++j)
        {
            for (int k = 0; k < ppedge.z; ++k)
            {
                glm::vec3 pos(i, j, k);
                pos *= dist;
                pos += min + prototype->getRadius();

                positions[id] = glm::vec4(pos, 1.0f);
                ++id;
            }
        }
    }

    return ppedge.x * ppedge.y * ppedge.z;
}

void SPHSystem::step(double deltaTime)  
{
    uint mid = prototype->getMesh()->getId();
    VAO_t vao = System::getRender()->getBufferObject(mid);
    size_t bytes;
    glm::vec4* positions;

    // pensar en encapsular las llamadas de openGL en la funcion solver->moveParticles()
    gpuErrchk(cudaGraphicsMapResources(1, &vao.cuda_id, 0)); // Map resources
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&positions, &bytes, vao.cuda_id)); // Get pointer of mapped data

    //solver->moveParticles(positions); // Run wrapped CUDA kernel
    //gpuErrchk(cudaDeviceSynchronize()); // Wait between each kernel?
    solver->computeDensity(positions);
    gpuErrchk(cudaDeviceSynchronize()); // Wait to complete kernel execution


    gpuErrchk(cudaGraphicsUnmapResources(1, &vao.cuda_id, 0)); // Unmap resources
}