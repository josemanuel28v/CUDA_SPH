#include "example.cuh"
#include "common.h"
#include "System.h"
#include "SPHSystem.h"
#include "CustomParticle.h"
#include "CameraKeyboard.h"
#include <iostream>

int main()
{
    //example2();
    
    System::init(640, 480);

    glm::vec3 position(0.0f, 0.0f, 2.0f);
    glm::vec3 up(0.0f, 1.0f, 0.0f);
    glm::vec3 lookAt(0.0f, 0.0f, 0.0f);
    Camera* camera = new CameraKeyboard(Camera::PERSPECTIVE, position, up, lookAt);

    CustomParticle* p = new CustomParticle(0.05f);
    glm::vec3 min(-1.0f, -1.0f, -1.0f);
    glm::vec3 max(1.0f, 1.0f, 1.0f);
    SPHSystem* sphSystem = new SPHSystem(min, max, p);

    System::setParticleSystem(sphSystem);
    System::setCamera(camera);
    System::mainLoop();
    System::releaseMemory();
}