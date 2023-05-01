#include "RenderProgram.h"
#include <fstream>
#include <sstream>

RenderProgram::RenderProgram()
{
	pid = glCreateProgram();
}

GLint RenderProgram::checkShaderError(GLuint shId)
{
	GLint success = 1;
	char* infoLog = new char[1024];
	infoLog[0] = '\0';
	
	glGetShaderiv(shId, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(shId, 1024, nullptr, infoLog);
		std::cout << "Error en shader\n" << infoLog << "\n";
		// exit(-1);
	}
	errorMSGs += "\n" + std::string(infoLog);
	
	return success;
}

GLint RenderProgram::checkProgramError()
{
	GLint program_linked;
	glGetProgramiv(pid, GL_LINK_STATUS, &program_linked);
	if (program_linked != GL_TRUE)
	{
		GLsizei log_length = 0;
		GLchar message[1024];
		glGetProgramInfoLog(pid, 1024, &log_length, message);
		std::cout << "Error en linkado de programa: \n" << message << "\n\n";
	}

	return program_linked;
}

std::string RenderProgram::readFile(const std::string fileName) 
{
	std::ifstream inFile(fileName);

	if (inFile.is_open())
	{
		std::stringstream strStream;
		strStream << inFile.rdbuf(); 
		std::string str = strStream.str(); 
		inFile.close();

		return str;
	}
	else
	{
		std::cout << "Error leyendo " << fileName << std::endl;
		return "";
	}
}

void RenderProgram::setupShaderVarList() {
	
	int count = 0;
	int bufSize = 100;
	char* name = new char[bufSize];
	GLenum type;
	int size = 0;
	int length = 0;

	glGetProgramiv(pid, GL_ACTIVE_ATTRIBUTES, &count);
	//std::cout << "Attributes: " << count << std::endl;
	for (int i = 0; i < count; i++)
	{
		glGetActiveAttrib(pid, (GLuint)i, bufSize, &length, &size, &type,name);
		vars[std::string(name)] = glGetAttribLocation(pid,name);
		//std::cout << "Attribute: " << std::string(name) << std::endl;
	}

	glGetProgramiv(pid, GL_ACTIVE_UNIFORMS, &count);
	//std::cout << "Uniforms: " << count << std::endl;
	for (int i = 0; i < count; i++)
	{
		glGetActiveUniform(pid, (GLuint)i, bufSize, &length, &size,&type, name);
		vars[std::string(name)] = glGetUniformLocation(pid,name);
		//std::cout << "Uniform: " << std::string(name) << std::endl;
	}
	std::cout << std::endl;

}

void RenderProgram::linkPrograms()
{
	use();
	for (auto& sh : shaders) 
	{
		glAttachShader(pid, sh.second);
	}
	glLinkProgram(pid);
	checkProgramError();
	setupShaderVarList();
}

std::string RenderProgram::getErrorMsg(GLuint shaderID)
{
	GLint success = 1;
	char* infoLog = new char[1024];
	infoLog[0] = '\0';
	glGetShaderiv(shaderID, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(shaderID, 1024, nullptr, infoLog);
		std::cout << "Error en shader\n" << infoLog << "\n";
		// exit(-1);
	}
	errorMSGs += "\n" + std::string(infoLog);
	return errorMSGs;
}

void RenderProgram::use()
{
	glUseProgram(pid);
}

void RenderProgram::setProgram(std::string fileName, RenderTypes type)
{
	
	std::string prgSource = readFile(fileName);
	
	if (type == RenderTypes::VERTEX)
	{
		this->shaders[type] = glCreateShader(GL_VERTEX_SHADER);
	}
	else if (type == RenderTypes::FRAGMENT)
	{
		this->shaders[type] =  glCreateShader(GL_FRAGMENT_SHADER);
	}

	const char* source = prgSource.c_str();
	GLuint shId = this->shaders[type];
	glShaderSource(shId, 1, &source, nullptr);
	glCompileShader(shId);

	checkShaderError(shId);
}

void RenderProgram::setInt(std::string loc, int val) 
{
	glUniform1i(vars[loc], val);
}

void RenderProgram::setFloat(std::string loc, float val) 
{
	glUniform1f(vars[loc], val);
}

void RenderProgram::setVec3(std::string loc, const glm::vec3& vec) 
{
	glUniform3fv(vars[loc], 1, glm::value_ptr(vec));
}

void RenderProgram::setVec4(std::string loc, const glm::vec4& vec) 
{
	glUniform4fv(vars[loc], 1, glm::value_ptr(vec));
}

void RenderProgram::setMatrix(std::string loc, const glm::mat4& matrix) 
{
	glUniformMatrix4fv(vars[loc], 1, GL_FALSE, glm::value_ptr(matrix));
}

void RenderProgram::setTexture2D(std::string loc, unsigned int textUnit)
{
	glUniform1i(vars[loc], textUnit);
}


