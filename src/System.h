#pragma once
#include "Render.h"
#include "InputManager.h"
#include "ParticleSystem.h"
#include "Camera.h"
#include <vector>

class TimeManager 
{
	const uint NUM_SAMPLES = 100;
	double newTime = 0;
	double deltaTime = 0;
	double meanDeltaTime = 0;
	double lastTime = 0;
	std::vector<double> samples;
	uint index = 0;

public:

	TimeManager()
	{
		samples.resize(NUM_SAMPLES);
	}

	void update() 
	{
		newTime = glfwGetTime();
		deltaTime = newTime - lastTime;
		lastTime = newTime;

		if (index < NUM_SAMPLES)
		{
			samples[index] = deltaTime;
			meanDeltaTime = deltaTime;
			index++;
		}
		else
		{
			meanDeltaTime = 0.0;
			for (uint i = 1; i < NUM_SAMPLES; ++i)
			{
				samples[i - 1] = samples[i]; 
				meanDeltaTime += samples[i];
			}
			samples[NUM_SAMPLES - 1] = deltaTime;
			meanDeltaTime += deltaTime;
			meanDeltaTime /= NUM_SAMPLES;
		}
	}

	double getDeltaTime()
	{
		return deltaTime;
	}

	double getMeanDeltaTime()
	{
		return meanDeltaTime;
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
	static ParticleSystem* getParticleSystem() { return psystem; }

protected:
    inline static bool exit;
	inline static bool pause;
	inline static bool drawBoundary;

    inline static glm::mat4 model;
    inline static Camera* camera = nullptr;
    inline static Render* render = nullptr;
    inline static InputManager* inputManager = nullptr;

    inline static std::vector<Object*> objects;
    inline static ParticleSystem* psystem = nullptr;

	static void events();
};