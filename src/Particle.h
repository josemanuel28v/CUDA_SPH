#pragma once
#include "Object.h"

class Particle : public Object
{
public:
    Particle() {}
    Particle(float radius) { this->radius = radius; }
    void setRadius(float radius) { this->radius = radius; }
    float getRadius() { return radius; }

private:

    float radius;
};