#pragma once
#define GLAD_ONLY_HEADERS
#include "common.h"
#include "types.h"

class Texture
{
public:

    struct color32_t 
    {
        unsigned char r, g, b, a;
    };

	enum TextureType
	{
		COLOR2D = 0, COLOR3D = 1, NORMAL = 2, DEPTH_FB = 3, COLOR_FB = 4
	};

	Texture() {}
    Texture(std::string fileName);
	void bind(uint);
	void load(std::string fileName);
	//void load(std::string left, std::string right, std::string front, std::string back, std::string top, std::string bottom);
	
	GLuint getId() const { return id; };
	glm::ivec2 getSize() const { return size; };
	TextureType getType() const { return type; }

	void setSize(glm::ivec2 size) { this->size = size; }
	void setType(TextureType type) { this->type = type; }

protected:
	TextureType type;
	GLuint id;
	glm::ivec2 size;
	inline static unsigned textUnitCounter = 0;
};