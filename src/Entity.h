#pragma once
#define GLAD_ONLY_HEADERS
#include "common.h"

class Entity
{
	glm::mat4 modelMt;

protected:

	glm::vec4 position;
	glm::vec4 rotation; 
	glm::vec4 scaling;  
	
public:

	Entity();
	virtual ~Entity() {}
	virtual void computeModelMatrix();
	virtual void step(double deltaTime) {};

	virtual void setPosition(glm::vec4 position);
	virtual void setRotation(glm::vec4 rotation);
	virtual void setScaling(glm::vec4 scaling);
	virtual void setModelMt(glm::mat4 modelMt);

	virtual glm::vec4 getPosition();
	virtual glm::vec4 getRotation();
	virtual glm::vec4 getScaling();
	virtual glm::mat4 getModelMt();
};

