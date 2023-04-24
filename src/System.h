#pragma once
#include "Render.h"
#include "InputManager.h"
#include "ParticleSystem.h"
#include "Camera.h"
#include <vector>

class TimeManager {

	double newTime = 0;
	double deltaTime = 0;
	double lastTime = 0;

public:

	void update() 
	{
		newTime = glfwGetTime();
		deltaTime = newTime - lastTime;
		lastTime = newTime;
	}

	double getDeltaTime()
	{
		return deltaTime;
	}

	double getTime()
	{
		return newTime;
	}
};

class System
{
public:
    static void init(uint width, uint height);
	static void mainLoop();
	static void releaseMemory();
    
    static void setParticleSystem(ParticleSystem* ps) { psystem = ps; }
    static void setCamera(Camera* camera) { System::camera = camera; }
    static void setModelMat(glm::mat4 model) { System::model = model;}

    static Camera* getCamera() { return camera; }
    static glm::mat4 getModelMat() { return model; }
	static InputManager* getInputManager() { return inputManager; }
	static Render* getRender() { return render; }

protected:
    inline static bool exit;

    inline static glm::mat4 model;
    inline static Camera* camera;
    inline static Render* render;
    inline static InputManager* inputManager;

    inline static std::vector<Object*> objects;
    inline static ParticleSystem* psystem;
};