#include "InputManager.h"
#include "types.h"

void InputManager::init()
{
    // Poner todas las teclas a false
    unsigned int numKeys = sizeof(keybEvent) / sizeof(bool);
    for (uint i = 0; i < numKeys; i++)
    {
        keybEvent[i] = false;
    }

    glfwSetKeyCallback(window, keyManager);
    glfwSetCursorPosCallback(window, mouseManager);
    glfwSetFramebufferSizeCallback(window, resizeManager);
    
    // Ocultar cursor y fijarlo en el centro de la pantalla
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPos(window, 0.0, 0.0);
}

bool InputManager::isPressed(int key)
{
    return keybEvent[key];
}

glm::ivec2 InputManager::getMousePosition()
{
    return glm::ivec2(xpos, ypos);
}

glm::ivec2 InputManager::getOldMousePosition()
{
    return glm::ivec2(oldxpos, oldypos);
}

void InputManager::keyManager(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    // Todas las keys que recoge son el may�scula
    switch (action)
    {
    case GLFW_PRESS:
        keybEvent[key] = true;
        break;

    case GLFW_RELEASE:
        keybEvent[key] = false;
        break;
    }
}

void InputManager::mouseManager(GLFWwindow* window, double xpos, double ypos) 
{
    oldxpos = InputManager::xpos;
    oldypos = InputManager::ypos;
    InputManager::xpos = (int)xpos;
    InputManager::ypos = (int)ypos;
}

void InputManager::resizeManager(GLFWwindow* window, int width, int height)
{
    InputManager::width = width;
    InputManager::height = height;
    resized = true;
}