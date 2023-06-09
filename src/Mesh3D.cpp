#include "Mesh3D.h"

Mesh3D::Mesh3D()
{
    id = counter++;

    vertices = new std::vector<vertex_t>();
	indices = new std::vector<glm::uint32>();
    material = nullptr;
}

Mesh3D::~Mesh3D()
{
    if (vertices) delete vertices;
    if (indices) delete indices;
    if (material) delete material;
}