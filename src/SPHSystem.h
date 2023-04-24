#pragma once
#include "ParticleSystem.h"
#include "SPH.cuh"


class SPHSystem : public ParticleSystem
{
public:

    SPHSystem(glm::vec3 min, glm::vec3 max, Particle* prototype);
    uint initFluidBlock(glm::vec3 min, glm::vec3 max);
    void step(double deltaTime) override;

    void setSmoothingLength(float h) { this->h = h; }
    void setMass(float mass) { this->mass = mass; }

private:
    // demás arrays a parte de posición

    SPH* solver;
    float h;
    float mass;

    std::vector<float> densities;

};