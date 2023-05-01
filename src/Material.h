#pragma once
#include "RenderProgram.h"
#include "Texture.h"

class Material
{
public:

    Material();
    ~Material();
    void setTexture(Texture* texture) { this->texture = texture; }
    void setTexturing(bool texturing) { this->texturing = texturing; }
    void setProgram(RenderProgram* program) { this->program = program; }

    Texture* getTexture() { return texture; }
    bool getTexturing() { return texturing; }
    RenderProgram* getProgram() { return program; }

    void prepare();
    void loadPrograms(std::vector<std::string> fileNames);

private:
    RenderProgram* program = nullptr;
    Texture* texture = nullptr;
    bool texturing;
};