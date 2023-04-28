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

class SPH
{
public:

   // CPU pointers
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
   void allocateCudaMemory(); // Reservará memoria en la GPU para los datos que no hayan sido ya instanciados por OpenGL (de momento OpenGL solo crea buffers para el array de posiciones)
   void freeCudaMemory();

   // SPH computation
   void checkValues();
   void reset(cudaGraphicsResource* positionBufferObject, glm::vec4* h_positions);
   void step(VAO_t positionBufferObject);

private:

   // Host pointers
   int* size;
   float* timeStep;
   float* h; 
   float* mass;
   float* density0;
   float* stiffness;
   float* viscosity;
   float* spikyConst;
   float* cubicConstK;
   float* radius;
   float* densities;
   float* pressures;
   glm::vec3* forces;
   glm::vec3* velocities;
   glm::vec3* minDomain;
   glm::vec3* maxDomain;

   // Device pointers
   int* d_size;
   float* d_timeStep;
   float* d_h;
   float* d_mass;
   float* d_density0;
   float* d_stiffness;
   float* d_viscosity;
   float* d_spikyConst;
   float* d_cubicConstK;
   float* d_radius;
   float* d_densities;
   float* d_pressures;
   glm::vec3* d_forces;
   glm::vec3* d_velocities;
   glm::vec3* d_minDomain;
   glm::vec3* d_maxDomain;

   // Grid stuff
   uint32_t* cellIndexBuffer;
   uint32_t* particleIndexBuffer;
   uint32_t* cellOffsetBuffer;

   uint32_t* d_cellIndexBuffer;
   uint32_t* d_particleIndexBuffer;
   uint32_t* d_cellOffsetBuffer;
};




