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
    SPHSystem* sphSystem = simpleDamBreak();
    //SPHSystem* sphSystem = doubleDamBreak();

    // System setup
    System::setParticleSystem(sphSystem);
    System::setCamera(camera);
    System::mainLoop();
    System::releaseMemory();
}

SPHSystem* simpleDamBreak()
{
    float prad = 0.023f; // 0.023 para 80000 
    CustomParticle* p = new CustomParticle(prad);
    glm::vec3 min(-1.0f, -1.0f + 2 * prad, -1.0f);
    glm::vec3 max(1.0f, 1.0f, 1.0f);
    glm::vec3 minDomain(-1.5f, -1.0f, -1.5f);
    glm::vec3 maxDomain(1.5f, 2.0f, 1.5f);

    SPHSystem* sphSystem = new SPHSystem(p);
    sphSystem->setFluid({{min, max}});
    sphSystem->setBoundary(minDomain, maxDomain);
    sphSystem->setStiffness(150.0f);
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
    glm::vec3 min(-1.584474503993988f, -1.5792612135410309f + 2 * prad, -1.5781463980674744f);
    glm::vec3 max(-0.4316301941871643f, 0.6732102334499359f, -0.42530208826065063f);
    fluid.push_back({min, max}); 
    min = glm::vec3(0.42687326669692993f, -1.5792612135410309f, 0.43262726068496704f);
    max = glm::vec3(1.5797175765037537f,  0.6732102334499359f, 1.5854715704917908f);
    fluid.push_back({min, max}); 

    SPHSystem* sphSystem = new SPHSystem(p);
    sphSystem->setFluid(fluid);
    sphSystem->setBoundary(glm::vec3(-1.600000023841858f, -1.603513240814209f, -1.600000023841858f),  glm::vec3(1.600000023841858f, 8.03462553024292f, 1.600000023841858f));
    sphSystem->setStiffness(250.0f);
    sphSystem->setViscosity(0.1f);
    sphSystem->setTimeStep(0.003f);
    sphSystem->setReferenceDensity(1000.0f);

    return sphSystem;
}



