#include "SPHSystem.h"
#include "types.h"
#include "VAO_t.h"
#include "System.h"

SPHSystem::SPHSystem(Particle* prototype)
{
    this->prototype = prototype;
    this->solver = new SPHSolver();
    this->size = 0;
    this->fluidSize = 0;
    this->boundarySize = 0;
}

void SPHSystem::init()
{    
    this->fluidSize = createFluid(fluid);
    this->boundarySize = createBoundary(minDomain, maxDomain);
    this->size = this->fluidSize + this->boundarySize;

    h = 4.0f * prototype->getRadius();
    mass = density0 * pow(2.0f * prototype->getRadius(), 3.0f); // m = d * v

    // Set host 
    solver->setSmoothingLength(&h);
    solver->setParticleRadius(prototype->getRadiusAddress());
    solver->setParticleMass(&mass);
    solver->setReferenceDensity(&density0);
    solver->setSize((int*)&size);
    solver->setFluidSize((int*)&fluidSize);
    solver->setBoundarySize((int*)&boundarySize);
    solver->setDensities(densities.data());
    solver->setPressures(pressures.data());
    solver->setForces(forces.data());
    solver->setVelocities(velocities.data());
    solver->setMinDomain(&minDomain);
    solver->setMaxDomain(&maxDomain);
    solver->setStiffness(&stiffness);
    solver->setViscosity(&viscosity);
    solver->setTimeStep(&timeStep);

    std::cout << "post set host pointers" << std::endl;

    // Set device pointers
    solver->init();

    std::cout << "post cudasolver init" << std::endl;

    // Print data
    std::cout << "------------------- SPH Fluid ------------------- " << std::endl;
    std::cout << "Number of fluid particles:     " << fluidSize << std::endl; 
    std::cout << "Number of boundary particles:  " << boundarySize << std::endl; 
    std::cout << "Stiffness:                     " << stiffness << std::endl; 
    std::cout << "Viscosity:                     " << viscosity << std::endl; 
    std::cout << "Particle radius:               " << prototype->getRadius() << std::endl; 
    std::cout << "Reference density:             " << density0 << std::endl; 
    std::cout << "Smoothing length:              " << h << std::endl;
    std::cout << "Particle mass:                 " << mass << std::endl;
    std::cout << "------------------------------------------------- " << std::endl;
}

void SPHSystem::prestep()
{
    uint mid = prototype->getMesh()->getId();
    VAO_t vao = System::getRender()->getBufferObject(mid);

    solver->precomputeBoundaryNeighbors(vao); 
}

void SPHSystem::reset() 
{
    uint mid = prototype->getMesh()->getId();
    VAO_t vao = System::getRender()->getBufferObject(mid);

    this->fluidSize = createFluid(fluid);
    this->boundarySize = createBoundary(minDomain, maxDomain);
    this->size = this->fluidSize + this->boundarySize;
    solver->reset(vao, &positions[0]);
}

void SPHSystem::setFluid(std::vector<Fluid> fluid) 
{ 
    this->fluid = fluid;
}

void SPHSystem::setBoundary(glm::vec3 min, glm::vec3 max) 
{ 
    this-> minDomain = min; 
    this->maxDomain = max; 
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

uint SPHSystem::createBoundary(glm::vec3 min, glm::vec3 max)
{
    float supportRadius = 4.0f * prototype->getRadius();

    // Longitud del cubo en cada eje
    glm::vec3 l = glm::abs(min - max);
    glm::ivec3 n = glm::ceil(l / (0.5f * supportRadius));

    unsigned numBParts = n.x * n.y * n.z;

    // De momento se redimensiona todo: mirar que propiedades no son utilizadas por las boundary particles
    positions.resize(this->fluidSize + numBParts);
    velocities.resize(this->fluidSize + numBParts);
    forces.resize(this->fluidSize + numBParts);
    pressures.resize(this->fluidSize + numBParts);
    densities.resize(this->fluidSize + numBParts);
    
    // Distancia entre particulas de cada eje
    glm::vec3 d = l / glm::vec3(n);

    unsigned id = this->fluidSize;
    for (int i = 0; i <= n.x; i++)
        for (int j = 0; j <= n.y; j++)
            for (int k = 0; k <= n.z; k++)
                if ((i == 0 || i == n.x) ||
                    (j == 0 || j == n.y) ||
                    (k == 0 || k == n.z))
                {
                    glm::vec3 position(i * d.x, j * d.y, k * d.z);
                    position += min;
                    
                    positions[id] = glm::vec4(position, 1.0f);
                    velocities[id] = glm::vec3(0.0f);
                    forces[id] = glm::vec3(0.0f);
                    ++id;
                }

    return numBParts;
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