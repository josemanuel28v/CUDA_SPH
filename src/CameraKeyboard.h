#pragma once
#include "Camera.h"

class CameraKeyboard : public Camera
{
public:
	CameraKeyboard(ProjectionType type, glm::vec3 position, glm::vec3 up, glm::vec3 lookAt);
	virtual void step(double deltaTime) override;
};

