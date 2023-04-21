#include "System.h"

void System::init(uint width, uint height)
{
    render = new Render(width, height);
    inputManager = new InputManager();

    render->init();
    inputManager->setWindow(render->getWindow());
    inputManager->init();
}

void System::mainLoop()
{

}

void System::exit()
{

}

void System::releaseMemory()
{
}
