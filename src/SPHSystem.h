#pragma once
#include "ParticleSystem.h"
#include "SPHSolver.cuh"

struct Fluid
{
    glm::vec3 min;
    glm::vec3 max;
};
 
class SPHSystem : public ParticleSystem
{
public:

    SPHSystem(Particle* prototype);
    ~SPHSystem() { if (solver) delete solver; }
    // Debe usarse en el orden: createFluid -> createBoundary
    uint createFluid(const std::vector<Fluid>& fluid);
    uint createBoundary(glm::vec3 min, glm::vec3 max);
    void init() override;
    void prestep() override;
    void step(double deltaTime) override;
    void reset() override;
    void release();

    void setSmoothingLength(float h) { this->h = h; }
    void setBoundary(glm::vec3 min, glm::vec3 max);
    void setStiffness(float stiffness) { this->stiffness = stiffness; }
    void setViscosity(float viscosity) { this->viscosity = viscosity; }
    void setReferenceDensity(float density0) { this->density0 = density0; }
    void setTimeStep(float timeStep) { this->timeStep = timeStep; }
    void setFluid(std::vector<Fluid> fluid);

    uint getDrawSize() override { return fluidSize; } 

private:

    uint fluidSize;
    uint boundarySize;

    SPHSolver* solver;

    float h;
    float mass;
    float density0;
    float stiffness;
    float viscosity;
    float timeStep;

    std::vector<float> densities;
    std::vector<float> pressures;
    std::vector<glm::vec3> velocities;
    std::vector<glm::vec3> forces;

    std::vector<Fluid> fluid;

    glm::vec3 minFluid;
    glm::vec3 maxFluid;
    glm::vec3 minDomain;
    glm::vec3 maxDomain;
};