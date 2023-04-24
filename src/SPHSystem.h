#pragma once
#include "ParticleSystem.h"

class SPHSystem : public ParticleSystem
{
public:

    SPHSystem(glm::vec3 min, glm::vec3 max, Particle* prototype);
    uint initFluidBlock(glm::vec3 min, glm::vec3 max);
    void step(double deltaTime) override;

private:
    // demás arrays a parte de posición
};