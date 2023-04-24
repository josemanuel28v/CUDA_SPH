#pragma once
#define GLAD_ONLY_HEADERS
#include "common.h"

struct vertex_t
{
	glm::vec4 position;
	glm::vec4 normal;
	glm::vec4 tangent;
	glm::vec2 textCoord;
};
