#include "Texture.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../extern/stb_image/stb_image.h"

Texture::Texture(std::string fileName)
{
    load(fileName);
}

void Texture::bind(uint textUnit)
{
    switch (this->type)
	{
	case COLOR2D:
		glActiveTexture(GL_TEXTURE0 + textUnit);
		glBindTexture(GL_TEXTURE_2D, id);
		break;

	case COLOR3D:
		glActiveTexture(GL_TEXTURE0 + textUnit);
		glBindTexture(GL_TEXTURE_CUBE_MAP, id);
		break;
	}
}

void Texture::load(std::string fileName)
{
    int comp = 0;
	color32_t* pixels = (color32_t*)stbi_load(fileName.c_str(), &size.x, &size.y, &comp, 4);

	glGenTextures(1, &id);
	glBindTexture(GL_TEXTURE_2D, id);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size.x, size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

	stbi_image_free(pixels);

	this->type = COLOR2D;
}