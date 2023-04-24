#pragma once
#include <unordered_map>
#include <string>
#define GLAD_ONLY_HEADERS
#include "common.h"
#include "types.h"

class RenderProgram
{
public:

	enum class RenderTypes {
		VERTEX = 0, FRAGMENT = 1
	};

	std::unordered_map<std::string, uint> vars;

	RenderProgram();
	~RenderProgram() {}
	void linkPrograms();
	std::string getErrorMsg(GLuint shaderID);
	void use();
	void setInt(std::string, int val);
	void setFloat(std::string loc, float val);
	void setVec3(std::string loc, const glm::vec3& vec);
	void setVec4(std::string loc, const glm::vec4& vec);
	void setMatrix(std::string loc, const glm::mat4& matrix);
	void setTexture2D(std::string loc, uint textUnit);

    void setProgram(std::string programSrc, RenderTypes type);

private:
	uint pid = 0;
    std::string errorMSGs;
    std::unordered_map<RenderTypes, uint> shaders;

    GLint checkShaderError(GLuint shId);
    GLint checkProgramError();
    std::string readFile(const std::string fileName);
    void setupShaderVarList();
};
