#include "common.h"
#include "System.h"
#include "SPHSystem.h"
#include "CustomParticle.h"
#include "CameraKeyboard.h"
#include <iostream>

int main()
{    
    System::init(640, 480);

    // Camera setup
    glm::vec3 position(0.0f, 0.0f, 4.0f);
    glm::vec3 up(0.0f, 1.0f, 0.0f);
    glm::vec3 lookAt(0.0f, 0.0f, 0.0f);
    Camera* camera = new CameraKeyboard(Camera::PERSPECTIVE, position, up, lookAt);

    // SPHSystem setup
    float prad = 0.043f;
    CustomParticle* p = new CustomParticle(prad);
    glm::vec3 min(-1.0f, -1.0f, -1.0f);
    glm::vec3 max(1.0f, 1.0f, 1.0f);
    glm::vec3 minDomain(-1.5f, -1.0f + 2 * prad, -1.5f);
    glm::vec3 maxDomain(1.5f, 2.0f, 1.5f);

    SPHSystem* sphSystem = new SPHSystem(p);
    sphSystem->setFluid(min, max);
    sphSystem->setDomain(minDomain, maxDomain);
    sphSystem->setStiffness(100.0f);
    sphSystem->setViscosity(0.1f);
    sphSystem->setTimeStep(0.01f);
    sphSystem->setReferenceDensity(1000.0f);

    // System setup
    System::setParticleSystem(sphSystem);
    System::setCamera(camera);
    System::mainLoop();
    System::releaseMemory();
}