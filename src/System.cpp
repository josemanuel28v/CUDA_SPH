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
    drawBoundary = false;
}

void System::mainLoop()
{
    TimeManager tm;

    for (auto obj: objects)
    {
        //render->setupObject();
    }

    if (psystem)
    {
        psystem->init();
        render->setupObject(psystem->getPrototype(), psystem->getSize(), psystem->getPositions());
        psystem->prestep();
    }

    while (!exit)
    {
        tm.update();

        inputManager->setWindowTitle("FPS: " + std::to_string(int(round(1.0f / tm.getMeanDeltaTime()))));

        render->clearDisplay();

        camera->step(tm.getDeltaTime());
        if (!pause) psystem->step(tm.getDeltaTime());

        uint psysSize = drawBoundary ? psystem->getSize() : psystem->getDrawSize();
        render->drawObject(psystem->getPrototype(), psysSize);
        render->swapBuffers();

        glfwPollEvents();

        events();
    }
}

void System::events()
{
    static bool pressed = false;
    static bool boundaryPressed = false;

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

    if (inputManager->isPressed('B') && !boundaryPressed)
    {
        drawBoundary = !drawBoundary;
        boundaryPressed = true;
    }
    else if (!inputManager->isPressed('B'))
    {
        boundaryPressed = false;
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

    inputManager->reset();
}

void System::releaseMemory()
{
    if (psystem) psystem->release();
    if (inputManager) delete inputManager;
    if (camera) delete camera;
    if (render) 
    {
        for (auto obj : objects)
        {
            render->removeObject(obj);
        }
        delete render;
    }
}
