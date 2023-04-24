#include "Entity.h"

Entity::Entity()
{
    position = glm::vec4(0.0f, 0.0f, 0.0f, 1.0);
    rotation = glm::vec4(0.0f, 0.0f, 0.0f, 1.0);
    scaling = glm::vec4(1.0f, 1.0f, 1.0f, 1.0);
    modelMt = glm::mat4(1.0f);
}

void Entity::computeModelMatrix()
{
    glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(position));
    model = glm::rotate(model, rotation.x, glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, rotation.y, glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, rotation.z, glm::vec3(0.0f, 0.0f, 1.0f));
    model = glm::scale(model, glm::vec3(scaling));
    modelMt = model;
}

void Entity::setPosition(glm::vec4 position)
{
    this->position = position;
}

void Entity::setRotation(glm::vec4 rotation)
{
    this->rotation = rotation;
}

void Entity::setScaling(glm::vec4 scaling)
{
    this->scaling = scaling;
}

void Entity::setModelMt(glm::mat4 modelMt)
{
    this->modelMt = modelMt;
}

glm::vec4 Entity::getPosition()
{
    return position;
}

glm::vec4 Entity::getRotation()
{
    return rotation;
}

glm::vec4 Entity::getScaling()
{
    return scaling;
}

glm::mat4 Entity::getModelMt()
{
    return modelMt;
}
