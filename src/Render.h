#pragma once
#define GLAD_ONLY_HEADERS
#include "common.h"
#include "Object.h"
#include "types.h"
#include "VAO_t.h"

class Render
{
public:
    Render(uint width, uint height);
    void init();
    void setupObject(Object* obj, unsigned numInstances, glm::vec4* positions);
    void removeObject(Object* obj);
    void drawObject(Object* obj, unsigned numInstances);
    void clearDisplay();
    void swapBuffers();
    bool isClosed();
    GLFWwindow* getWindow() { return window; }

    VAO_t getBufferObject(uint meshId) 
    { 
        if (bufferObjects.find(meshId) != bufferObjects.end())
        {
            return bufferObjects[meshId]; 
        }
        
        VAO_t vao;
        return vao;
    }

private:

    std::unordered_map<uint, VAO_t> bufferObjects;
    GLFWwindow* window;
    uint width, height;
};