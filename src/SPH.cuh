#include <cuda_runtime.h>
#include <glm/glm.hpp>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

class SPH
{
public:

   // CPU pointers
   void registerHostPointers(float* h, float *radius, float* mass, int* size, float* densities);

   // GPU Pointers
   void allocateCudaMemory(); // Reservará memoria en la GPU para los datos que no hayan sido ya instanciados por OpenGL (de momento OpenGL solo crea buffers para el array de posiciones)
   void freeCudaMemory();

   void moveParticles(glm::vec4* positions); // test function

   void computeDensity(glm::vec4* positions);
   void computePressureForce(glm::vec4* forces);
   void computeViscosityForce(glm::vec4* forces);
   void integration(glm::vec4* positions);

private:

   // Host pointers
   int* size;
   float* h; 
   float* mass;
   float* poly6Const;
   float* radius;
   float* densities;

   // Device pointers
   int* d_size;
   float* d_h;
   float* d_mass;
   float* d_poly6Const;
   float* d_radius;
   float* d_densities;
};


