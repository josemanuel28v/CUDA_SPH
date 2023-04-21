#pragma once
#include "Render.h"
#include "InputManager.h"
#include <vector>

class System
{
public:
    static void init(uint width, uint height);
	static void mainLoop();
	static void exit();
	static void releaseMemory();

protected:
    inline static Render* render;
    inline static InputManager* inputManager;
    inline static std::vector<Object*> objects;
};