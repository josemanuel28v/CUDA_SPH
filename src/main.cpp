#include "common.h"
#include "System.h"
#include "SPHSystem.h"
#include "CustomParticle.h"
#include "CameraKeyboard.h"
#include <iostream>

SPHSystem* simpleDamBreak();
SPHSystem* doubleDamBreak();

int main()
{    
    System::init(640, 480);

    // Camera setup
    glm::vec3 position(0.0f, 0.0f, 4.0f);
    glm::vec3 up(0.0f, 1.0f, 0.0f);
    glm::vec3 lookAt(0.0f, 0.0f, 0.0f);
    Camera* camera = new CameraKeyboard(Camera::PERSPECTIVE, position, up, lookAt);

    // SPHSystem setup
    //SPHSystem* sphSystem = simpleDamBreak();
    SPHSystem* sphSystem = doubleDamBreak();

    // System setup
    System::setParticleSystem(sphSystem);
    System::setCamera(camera);
    System::mainLoop();
    System::releaseMemory();
}

SPHSystem* simpleDamBreak()
{
    float prad = 0.023f;
    CustomParticle* p = new CustomParticle(prad);
    glm::vec3 min(-1.0f, -1.0f, -1.0f);
    glm::vec3 max(1.0f, 1.0f, 1.0f);
    glm::vec3 minDomain(-1.5f, -1.0f + 2 * prad, -1.5f);
    glm::vec3 maxDomain(1.5f, 2.0f, 1.5f);

    SPHSystem* sphSystem = new SPHSystem(p);
    sphSystem->setFluid({{min, max}});
    sphSystem->setDomain(minDomain, maxDomain);
    sphSystem->setStiffness(100.0f);
    sphSystem->setViscosity(0.1f);
    sphSystem->setTimeStep(0.005f);
    sphSystem->setReferenceDensity(1000.0f);

    return sphSystem;
}

SPHSystem* doubleDamBreak()
{
    float prad = 0.021f;
    CustomParticle* p = new CustomParticle(prad);
    std::vector<Fluid> fluid;
    glm::vec3 min(-1.584474503993988, -1.5792612135410309, -1.5781463980674744);
    glm::vec3 max(-0.4316301941871643, 0.6732102334499359, -0.42530208826065063);
    fluid.push_back({min, max}); 
    min = glm::vec3(0.42687326669692993, -1.5792612135410309, 0.43262726068496704);
    max = glm::vec3(1.5797175765037537,  0.6732102334499359, 1.5854715704917908);
    fluid.push_back({min, max}); 

    SPHSystem* sphSystem = new SPHSystem(p);
    sphSystem->setFluid(fluid);
    sphSystem->setDomain(glm::vec3(-1.600000023841858, -1.603513240814209, -1.600000023841858),  glm::vec3(1.600000023841858, 8.03462553024292, 1.600000023841858));
    sphSystem->setStiffness(100.0f);
    sphSystem->setViscosity(0.1f);
    sphSystem->setTimeStep(0.004f);
    sphSystem->setReferenceDensity(1000.0f);

    return sphSystem;
}



