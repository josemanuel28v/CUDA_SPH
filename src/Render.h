#pragma once
#include "common.h"
#include "Object.h"
#include "types.h"

class Render
{
public:
    Render(uint width, uint height);
    void init();
    void setupObject(Object *obj, unsigned numInstances);
    void removeObject(Object *obj, unsigned numInstances);
    void drawObject(Object *obj, unsigned numInstances);
    void clearDisplay();
    void swapBuffers();
    GLFWwindow* getWindow() { return window; }
private:
    GLFWwindow* window;
    uint width, height;
};