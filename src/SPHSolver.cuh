#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <vector>
#include "VAO_t.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

class SPHSolver
{
public:

   // Initialize and set CPU pointers
   void init();
   void setSmoothingLength(float* h) { this->h = h; }
   void setParticleRadius(float* radius) { this->radius = radius; }
   void setParticleMass(float* mass) { this->mass = mass; }
   void setReferenceDensity(float* density0) { this->density0 = density0; }
   void setNumParticles(int* size) { this->size = size; }
   void setDensities(float* densities) { this->densities = densities; }
   void setPressures(float* pressures) { this->pressures = pressures; }
   void setForces(glm::vec3* forces) { this->forces = forces; }
   void setVelocities(glm::vec3* velocities) { this-> velocities = velocities; }
   void setMinDomain(glm::vec3* minDomain) { this->minDomain = minDomain; }
   void setMaxDomain(glm::vec3* maxDomain) { this->maxDomain = maxDomain; }
   void setStiffness(float* stiffness) { this->stiffness = stiffness; }
   void setViscosity(float* viscosity) { this->viscosity = viscosity; }
   void setTimeStep(float* timeStep) { this->timeStep = timeStep; }

   // GPU Pointers
   void allocateCudaMemory(); 
   void freeCudaMemory();

   // SPH computation
   void reset(cudaGraphicsResource* positionBufferObject, glm::vec4* h_positions);
   void step(VAO_t positionBufferObject);
   void release();

private:

   // Host pointers
   int* size = nullptr;
   float* timeStep = nullptr;
   float* h = nullptr; 
   float* mass = nullptr;
   float* density0 = nullptr;
   float* stiffness = nullptr;
   float* viscosity = nullptr;
   float* spikyConst = nullptr;
   float* cubicConstK = nullptr;
   float* radius = nullptr;
   float* densities = nullptr;
   float* pressures = nullptr;
   glm::vec3* forces = nullptr;
   glm::vec3* velocities = nullptr;
   glm::vec3* minDomain = nullptr;
   glm::vec3* maxDomain = nullptr;

   // Grid
   uint32_t* cellIndexBuffer = nullptr;
   uint32_t* particleIndexBuffer = nullptr;
   uint32_t* cellOffsetBuffer = nullptr;

   // Device pointers
   int* d_size = nullptr;
   float* d_timeStep = nullptr;
   float* d_h = nullptr;
   float* d_mass = nullptr;
   float* d_density0 = nullptr;
   float* d_stiffness = nullptr;
   float* d_viscosity = nullptr;
   float* d_spikyConst = nullptr;
   float* d_cubicConstK = nullptr;
   float* d_radius = nullptr;
   float* d_densities = nullptr;
   float* d_pressures = nullptr;
   glm::vec3* d_forces = nullptr;
   glm::vec3* d_velocities = nullptr;
   glm::vec3* d_minDomain = nullptr;
   glm::vec3* d_maxDomain = nullptr;

   // Grid
   uint32_t* d_cellIndexBuffer = nullptr;
   uint32_t* d_particleIndexBuffer = nullptr;
   uint32_t* d_cellOffsetBuffer = nullptr;
};




