#pragma once
#include "Mesh3D.h"
#include "Entity.h"

class Object : public Entity
{
public:
    virtual Mesh3D* getMesh() { return mesh; }
    virtual void setMesh(Mesh3D* mesh) { this->mesh = mesh; }
    virtual ~Object() { if (mesh) delete mesh; }

private:
    Mesh3D* mesh = nullptr;
};