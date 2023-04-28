#include "SPHSystem.h"
#include "types.h"
#include "VAO_t.h"
#include "System.h"

SPHSystem::SPHSystem(Particle* prototype)
{
    this->prototype = prototype;
    this->solver = new SPH();
}

void SPHSystem::init()
{
    h = 4.0f * prototype->getRadius();
    mass = density0 * pow(2.0f * prototype->getRadius(), 3.0); // m = d * v

    // Set host and devices pointers 
    solver->setSmoothingLength(&h);
    solver->setParticleRadius(prototype->getRadiusAddress());
    solver->setParticleMass(&mass);
    solver->setReferenceDensity(&density0);
    solver->setNumParticles((int*)&size);
    solver->setDensities(densities.data());
    solver->setPressures(pressures.data());
    solver->setForces(forces.data());
    solver->setVelocities(velocities.data());
    solver->setMinDomain(&minDomain);
    solver->setMaxDomain(&maxDomain);
    solver->setStiffness(&stiffness);
    solver->setViscosity(&viscosity);
    solver->setTimeStep(&timeStep);

    // Set device pointers
    solver->init();

    // Print data
    std::cout << "----------- SPH Fluid ----------- " << std::endl;
    std::cout << "Number of particles:  " << size << std::endl; 
    std::cout << "Stiffness:            " << stiffness << std::endl; 
    std::cout << "Viscosity:            " << viscosity << std::endl; 
    std::cout << "Particle radius:      " << prototype->getRadius() << std::endl; 
    std::cout << "Reference density:    " << density0 << std::endl; 
    std::cout << "Smoothing length:     " << h << std::endl;
    std::cout << "Particle mass:        " << mass << std::endl;
    std::cout << "--------------------------------- " << std::endl;
}

void SPHSystem::reset() 
{
    uint mid = prototype->getMesh()->getId();
    VAO_t vao = System::getRender()->getBufferObject(mid);

    createFluid(minFluid, maxFluid);
    solver->reset(vao.cuda_p_id, &positions[0]);
}

uint SPHSystem::createFluid(glm::vec3 min, glm::vec3 max)
{
    float dist = 2.0f * prototype->getRadius();

    glm::ivec3 ppedge = floor((max - min) / dist);// + 1e-5f);
    uint numParticles = ppedge.x * ppedge.y * ppedge.z;
    
    positions.resize(numParticles);
    velocities.resize(numParticles);
    forces.resize(numParticles);
    pressures.resize(numParticles);
    densities.resize(numParticles);

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
                velocities[id] = glm::vec3(0.0f);
                forces[id] = glm::vec3(0.0f);
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

    solver->step(vao);
}

void SPHSystem::release()
{
    solver->freeCudaMemory(); 
}