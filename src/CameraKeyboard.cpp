#include "CameraKeyboard.h"
#include "System.h"

CameraKeyboard::CameraKeyboard(ProjectionType type, glm::vec3 position, glm::vec3 up, glm::vec3 lookAt) : Camera(type, position, up, lookAt)
{

}

void CameraKeyboard::step(double deltaTime)
{
	static glm::ivec2 oldMousePos(0, 0);

	glm::ivec2 mousePos = System::getInputManager()->getMousePosition();
	glm::vec3 movement(0.0f);

	float moveSpeed = 2.0f * deltaTime;
	float rotSpeed =  0.5f * deltaTime;

	// Update rotation with mouse movement
	rotation.x +=  (oldMousePos.y - mousePos.y) * rotSpeed;
	rotation.y += -(oldMousePos.x - mousePos.x) * rotSpeed;

	oldMousePos = mousePos;

	// X rotation constraint (-90, 90)
	if		(rotation.x > glm::radians( 89.0f)) rotation.x = glm::radians( 89.0f);
	else if (rotation.x < glm::radians(-89.0f)) rotation.x = glm::radians(-89.0f);

	updateCameraVectors();

	// Recalculate lookAt with new forward direction
	lookAt = glm::vec3(position) + forward * glm::length(lookAt - glm::vec3(position));

	// Keyboard
	if (System::getInputManager()->isPressed(GLFW_KEY_UP))    movement +=  moveSpeed * forward;
	if (System::getInputManager()->isPressed(GLFW_KEY_DOWN))  movement += -moveSpeed * forward;

	if (System::getInputManager()->isPressed(GLFW_KEY_LEFT))  movement += -moveSpeed * lateral;
	if (System::getInputManager()->isPressed(GLFW_KEY_RIGHT)) movement +=  moveSpeed * lateral;

	if (System::getInputManager()->isPressed('Q'))			 movement +=  moveSpeed * up;
	if (System::getInputManager()->isPressed('A'))			 movement += -moveSpeed * up;

	// Update position and lookAt with keyboard movement
	position += glm::vec4(movement, 0.0f);
	lookAt += movement;

	computeViewMatrix();
}
