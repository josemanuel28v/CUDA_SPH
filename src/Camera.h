#pragma once
#define GLAD_ONLY_HEADERS
#include "common.h"
#include "Entity.h"

class Camera : public Entity
{
public:

	enum ProjectionType
	{
		PERSPECTIVE,
		ORTOGRAPHIC
	};

	Camera(ProjectionType type, glm::vec3 position, glm::vec3 up, glm::vec3 lookAt);
	virtual glm::mat4 getProjection();
	virtual glm::mat4 getView();
	virtual void computeProjectionMatrix();
	virtual void computeViewMatrix();
	virtual void step(double timeStep) {};
	virtual void setFov(float fov);
	virtual void setAspect(float aspect);
	virtual void setNear(float near);
	virtual void setFar(float far);
	virtual void setLeft(float left);
	virtual void setRight(float right);
	virtual void setBottom(float bottom);
	virtual void setTop(float top);

protected:
	virtual void updateCameraVectors();

	ProjectionType type;

	glm::mat4 view;
	glm::mat4 projection;

	glm::vec3 forward;
	glm::vec3 lateral;
	glm::vec3 worldUp;
	glm::vec3 up;

	glm::vec3 lookAt;

	float fov;
	float aspect;
	float near;
	float far;

	float left;
	float right;
	float bottom;
	float top;
};

