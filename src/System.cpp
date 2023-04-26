#include "System.h"

void System::init(uint width, uint height)
{
    render = new Render(width, height);
    inputManager = new InputManager();

    render->init();
    inputManager->setWindow(render->getWindow());
    inputManager->init();

    exit = false;
    pause = false;
}

#include <iostream>
void System::mainLoop()
{
    TimeManager tm;
    bool pressed = false;

    for (auto obj: objects)
    {
        //render->drawObject();
    }

    if (psystem)
    {
        render->setupObject(psystem->getPrototype(), psystem->getSize(), psystem->getPositions());
        psystem->init();
    }

    while (!exit)
    {
        tm.update();

        render->clearDisplay();

        camera->step(tm.getDeltaTime());
        if (!pause) psystem->step(tm.getDeltaTime());

        render->drawObject(psystem->getPrototype(), psystem->getSize(), psystem->getPositions());
        render->swapBuffers();

        glfwPollEvents();

        if (inputManager->isPressed('E') || render->isClosed())
        {
            exit = true;
        }

        if (inputManager->isPressed('P') && !pressed)
        {
            pause = !pause;
            pressed = true;
        }
        else if (!inputManager->isPressed('P'))
        {
            pressed = false;
        }

        if (inputManager->isPressed('R'))
        {
            psystem->reset(); // Resetear posiciones en CPU y GPU
        }

        if (inputManager->isWindowResized())
		{
			glm::ivec2 size = inputManager->getWindowSize();
			glViewport(0, 0, size.x, size.y);
			inputManager->setWindowResized();
			camera->setAspect((float)size.x / size.y);
			camera->computeProjectionMatrix();
		}	
    }
}

void System::releaseMemory()
{
    if (psystem) psystem->release();
}
