#include "SPHSystem.h"
#include "types.h"
#include "VAO_t.h"
#include "System.h"

SPHSystem::SPHSystem(Particle* prototype)
{
    this->prototype = prototype;
    this->solver = new SPHSolver();
}

void SPHSystem::init()
{
    h = 4.0f * prototype->getRadius();
    mass = density0 * pow(2.0f * prototype->getRadius(), 3.0f); // m = d * v

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

    createFluid(fluid);
    solver->reset(vao.cuda_p_id, &positions[0]);
}

uint SPHSystem::createFluid(const std::vector<Fluid>& fluid)
{
    float dist = 2.0f * prototype->getRadius();
    std::vector<glm::ivec3> ppedge(fluid.size());

    uint numParticles = 0;
    for (uint fid = 0; fid < fluid.size(); ++fid)
    {
        ppedge[fid] = floor((fluid[fid].max - fluid[fid].min) / dist);
        numParticles += ppedge[fid].x * ppedge[fid].y * ppedge[fid].z;
    }
    
    positions.resize(numParticles);
    velocities.resize(numParticles);
    forces.resize(numParticles);
    pressures.resize(numParticles);
    densities.resize(numParticles);

    uint id = 0;

    for (uint fid = 0; fid < fluid.size(); ++fid)
    {
        for (int i = 0; i < ppedge[fid].x; ++i)
        {
            for (int j = 0; j < ppedge[fid].y; ++j)
            {
                for (int k = 0; k < ppedge[fid].z; ++k)
                {
                    glm::vec3 pos(i, j, k);
                    pos *= dist;
                    pos += fluid[fid].min + prototype->getRadius();

                    positions[id] = glm::vec4(pos, 1.0f);
                    velocities[id] = glm::vec3(0.0f);
                    forces[id] = glm::vec3(0.0f);
                    ++id;
                }
            }
        }
    }

    return numParticles;
}

void SPHSystem::step(double deltaTime)  
{
    uint mid = prototype->getMesh()->getId();
    VAO_t vao = System::getRender()->getBufferObject(mid);

    solver->step(vao);
}

void SPHSystem::release()
{
    solver->release(); 
}