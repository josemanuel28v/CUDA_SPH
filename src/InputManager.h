#pragma once
#define GLAD_ONLY_HEADERS
#include "common.h"

class InputManager
{
public:

	void init();
	bool isPressed(int key);
	glm::ivec2 getMousePosition();
	glm::ivec2 getOldMousePosition();

	glm::ivec2 getWindowSize() { return glm::ivec2(width, height); }
	bool isWindowResized() { return resized; }
	void setWindowResized() { resized = false; }
    void setWindow(GLFWwindow* window) { this->window = window; }

    static void keyManager(GLFWwindow* window, int key, int scancode, int action, int mods);
	static void mouseManager(GLFWwindow* window, double xpos, double ypos);
	static void resizeManager(GLFWwindow* window, int width, int height);

protected:

	inline static bool keybEvent[512];
	inline static int xpos, ypos, oldxpos, oldypos = 0;
	inline static int width, height;
	inline static bool resized = false;
    inline static GLFWwindow* window;
};