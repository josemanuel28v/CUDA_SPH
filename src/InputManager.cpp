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
    glfwSetMouseButtonCallback(window, mouseButtonManager);
    glfwSetFramebufferSizeCallback(window, resizeManager);
    //glfwSetScrollCallback(window, mouseScrollManager);
}

void InputManager::reset()
{
    scrollOffset = 0.0;
}

void InputManager::disableCursor()
{
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    //glfwSetCursorPos(window, 0.0, 0.0);
}

void InputManager::enableCursor()
{
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
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
    // Todas las keys que recoge son el mayuscula
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

void InputManager::mouseButtonManager(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        switch (action)
        {
            case GLFW_PRESS:
                rightButtonState = 1;
                break;

            case GLFW_RELEASE:
                rightButtonState = 0;
                break;
        } 
    }
}

void InputManager::resizeManager(GLFWwindow* window, int width, int height)
{
    InputManager::width = width;
    InputManager::height = height;
    resized = true;
}

void InputManager::mouseScrollManager(GLFWwindow* window, double xoffset, double yoffset)
{
    scrollOffset = yoffset;
}