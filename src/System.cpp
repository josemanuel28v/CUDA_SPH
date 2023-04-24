#include "System.h"

void System::init(uint width, uint height)
{
    render = new Render(width, height);
    inputManager = new InputManager();

    render->init();
    inputManager->setWindow(render->getWindow());
    inputManager->init();

    exit = false;
}

#include <iostream>
void System::mainLoop()
{
    TimeManager tm;

    for (auto obj: objects)
    {
        //render->drawObject();
    }

    if (psystem)
        render->setupObject(psystem->getPrototype(), psystem->getSize());

    while (!exit)
    {
        tm.update();

        render->clearDisplay();

        camera->step(tm.getDeltaTime());
        psystem->step(tm.getDeltaTime());

        render->drawObject(psystem->getPrototype(), psystem->getSize(), psystem->getPositions());
        render->swapBuffers();

        glfwPollEvents();

        if (inputManager->isPressed('E') || render->isClosed())
        {
            exit = true;
        }
    }
}

void System::releaseMemory()
{
}
