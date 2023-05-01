#pragma once
#include "vertex_t.h"
#include "Material.h"
#define GLAD_ONLY_HEADERS
#include "common.h"

class Mesh3D
{
public:
    Mesh3D();
    ~Mesh3D();
    Material* getMaterial() { return material; }
    std::vector<vertex_t>* getVertices() { return vertices; }
    std::vector<glm::uint32>* getIndices() { return indices; }
    uint getId() { return id; }
    
    void setMaterial(Material* material) { this->material = material; }
private:
    uint id;
    Material* material = nullptr;
    std::vector<vertex_t>* vertices = nullptr;
    std::vector<glm::uint32>* indices = nullptr;
    inline static uint counter = 0;
};