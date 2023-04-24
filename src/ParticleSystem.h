#pragma once
#define GLAD_ONLY_HEADERS
#include "common.h"
#include "Particle.h"
#include "types.h"

class ParticleSystem
{
public:

    ParticleSystem() {}
    ParticleSystem(uint size) 
    {
        prototype = nullptr;
        this->size = size;
    }

    virtual void step(double deltaTime) = 0;

    virtual Particle* getPrototype() { return prototype; }
    virtual uint getSize() { return size; }

    virtual void setPrototype(Particle* prototype) { this->prototype = prototype; }

    glm::vec4* getPositions() { return positions.data(); }

protected:

    Particle* prototype = nullptr;

    std::vector<glm::vec4> positions;
    uint size;
    // arrays of pos, vel, dens, forces ...
};